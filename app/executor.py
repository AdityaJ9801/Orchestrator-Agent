"""
Parallel execution engine.

Algorithm
---------
1. Build a dependency map from the TaskGraph.
2. Repeatedly find tasks whose all dependencies are COMPLETED.
3. Run an asyncio.gather over that "ready" batch.
4. If a dependency is FAILED / SKIPPED, mark dependents as SKIPPED.
5. Repeat until all tasks are settled.

Each agent call POSTs the task payload to /run on the target agent service.
On timeout the task is marked FAILED and execution continues (partial results).
"""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any, Callable, Coroutine, Dict, List, Optional

import httpx

from app.config import get_settings
from app.models import AgentName, TaskGraph, TaskNode, TaskResult, TaskStatus

logger = logging.getLogger(__name__)


# ── Agent dispatcher ─────────────────────────────────────────────────────────

async def _call_agent(
    *,
    agent: AgentName,
    url: str,
    payload: Dict[str, Any],
    timeout: float,
    client: httpx.AsyncClient,
) -> Any:
    """POST payload to <url>/run and return the JSON response body."""
    resp = await client.post(
        f"{url}/run",
        json=payload,
        timeout=timeout,
    )
    resp.raise_for_status()
    return resp.json()


# ── Core execution engine ────────────────────────────────────────────────────

async def execute_graph(
    graph: TaskGraph,
    *,
    on_task_start: Optional[Callable[[TaskResult], Coroutine]] = None,
    on_task_done:  Optional[Callable[[TaskResult], Coroutine]] = None,
    http_client:   Optional[httpx.AsyncClient] = None,
) -> List[TaskResult]:
    """
    Execute the TaskGraph with maximum parallelism.

    :param on_task_start: async hook called the moment a task begins.
    :param on_task_done:  async hook called when a task finishes (any status).
    :param http_client:   injectable httpx client (for testing with MockTransport).
    """
    settings = get_settings()
    registry = settings.agent_registry
    timeout  = settings.agent_timeout_seconds

    # Build result map (task_id → TaskResult)
    results: Dict[str, TaskResult] = {
        t.task_id: TaskResult(task_id=t.task_id, agent=t.agent)
        for t in graph.tasks
    }
    task_map: Dict[str, TaskNode] = {t.task_id: t for t in graph.tasks}

    # ── helpers ──────────────────────────────────────────────────────────────

    def _settled(tid: str) -> bool:
        return results[tid].status in (
            TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.SKIPPED
        )

    def _all_settled() -> bool:
        return all(_settled(tid) for tid in results)

    def _ready_tasks() -> List[TaskNode]:
        """Tasks that are PENDING and whose every dep is COMPLETED."""
        ready = []
        for t in graph.tasks:
            r = results[t.task_id]
            if r.status != TaskStatus.PENDING:
                continue
            if all(results[dep].status == TaskStatus.COMPLETED for dep in t.depends_on):
                ready.append(t)
        return ready

    def _should_skip(task: TaskNode) -> bool:
        """Skip task if any dependency is FAILED or SKIPPED."""
        return any(
            results[dep].status in (TaskStatus.FAILED, TaskStatus.SKIPPED)
            for dep in task.depends_on
        )

    # ── execution ─────────────────────────────────────────────────────────────

    own_client = http_client is None
    client = http_client or httpx.AsyncClient()

    try:
        while not _all_settled():
            # Mark tasks whose deps failed as SKIPPED
            for t in graph.tasks:
                r = results[t.task_id]
                if r.status == TaskStatus.PENDING and _should_skip(t):
                    r.status   = TaskStatus.SKIPPED
                    r.error    = "Skipped: upstream dependency failed"
                    r.ended_at = datetime.now(tz=timezone.utc)
                    if on_task_done:
                        await on_task_done(r)

            ready = _ready_tasks()
            if not ready:
                if _all_settled():
                    break
                # Guard against infinite loop (circular deps)
                logger.error("No ready tasks but not all settled — possible cycle")
                for t in graph.tasks:
                    r = results[t.task_id]
                    if r.status == TaskStatus.PENDING:
                        r.status = TaskStatus.FAILED
                        r.error  = "Execution stalled — possible dependency cycle"
                break

            async def _run_one(task: TaskNode) -> None:
                r = results[task.task_id]
                r.status     = TaskStatus.RUNNING
                r.started_at = datetime.now(tz=timezone.utc)

                if on_task_start:
                    await on_task_start(r)

                agent_url = registry.get(task.agent.value)
                if not agent_url:
                    r.status   = TaskStatus.FAILED
                    r.error    = f"Unknown agent: {task.agent.value}"
                    r.ended_at = datetime.now(tz=timezone.utc)
                    if on_task_done:
                        await on_task_done(r)
                    return

                # Build enriched payload: include results from dependencies
                enriched = dict(task.payload)
                enriched["_context"] = {
                    dep: results[dep].result
                    for dep in task.depends_on
                    if results[dep].status == TaskStatus.COMPLETED
                }

                try:
                    result = await _call_agent(
                        agent=task.agent,
                        url=agent_url,
                        payload=enriched,
                        timeout=timeout,
                        client=client,
                    )
                    r.result   = result
                    r.status   = TaskStatus.COMPLETED
                except httpx.TimeoutException:
                    logger.warning("Agent %s timed out for task %s", task.agent, task.task_id)
                    r.status = TaskStatus.FAILED
                    r.error  = f"Agent {task.agent.value} timed out after {timeout}s"
                except httpx.HTTPStatusError as exc:
                    logger.warning("Agent %s HTTP %s for task %s", task.agent, exc.response.status_code, task.task_id)
                    r.status = TaskStatus.FAILED
                    r.error  = f"HTTP {exc.response.status_code}: {exc.response.text[:200]}"
                except Exception as exc:
                    logger.exception("Unexpected error calling agent %s", task.agent)
                    r.status = TaskStatus.FAILED
                    r.error  = str(exc)
                finally:
                    r.ended_at = datetime.now(tz=timezone.utc)
                    if on_task_done:
                        await on_task_done(r)

            await asyncio.gather(*[_run_one(t) for t in ready])

    finally:
        if own_client:
            await client.aclose()

    return list(results.values())
