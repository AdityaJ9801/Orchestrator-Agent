"""
Integration tests for the parallel execution engine using httpx.MockTransport.

Scenarios tested
----------------
1. All agents succeed → all tasks COMPLETED.
2. An agent times out → task FAILED, downstream tasks SKIPPED.
3. Max parallelism: tasks with no shared deps run concurrently.
4. Dependency chain: sequential execution when depends_on is set.
5. Unknown agent in graph → task FAILED, does not crash engine.
6. on_task_start and on_task_done hooks are called exactly once per task.
"""
from __future__ import annotations

import asyncio
import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from app.executor import execute_graph
from app.models import AgentName, TaskGraph, TaskNode, TaskResult, TaskStatus

# ── Fixtures ──────────────────────────────────────────────────────────────────


def _graph(*nodes: TaskNode) -> TaskGraph:
    return TaskGraph(intent="test", tasks=list(nodes))


def _node(
    task_id: str,
    agent: AgentName = AgentName.SQL,
    depends_on: list[str] | None = None,
    payload: dict | None = None,
) -> TaskNode:
    return TaskNode(
        task_id=task_id,
        agent=agent,
        description=f"Task {task_id}",
        payload=payload or {},
        depends_on=depends_on or [],
    )


def _ok_transport(result: Any = {"ok": True}) -> httpx.MockTransport:
    """A transport that always returns HTTP 200 with *result* as JSON."""

    def _handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=result)

    return httpx.MockTransport(_handler)


def _timeout_transport() -> httpx.MockTransport:
    """A transport that always raises ReadTimeout."""

    def _handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("timed out", request=request)

    return httpx.MockTransport(_handler)


# ── Patch settings so we have proper URLs ────────────────────────────────────

MOCK_REGISTRY = {
    "context": "http://context-agent:8001",
    "sql":     "http://sql-agent:8002",
    "viz":     "http://viz-agent:8003",
    "ml":      "http://ml-agent:8004",
    "nlp":     "http://nlp-agent:8005",
    "report":  "http://report-agent:8006",
}


def _patch_registry():
    mock_settings = MagicMock()
    mock_settings.agent_registry     = MOCK_REGISTRY
    mock_settings.agent_timeout_seconds = 30.0
    return patch("app.executor.get_settings", return_value=mock_settings)


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Happy path — all agents succeed
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_all_tasks_complete():
    graph = _graph(
        _node("t1", AgentName.CONTEXT),
        _node("t2", AgentName.SQL,    depends_on=["t1"]),
        _node("t3", AgentName.REPORT, depends_on=["t2"]),
    )
    client = httpx.AsyncClient(transport=_ok_transport({"rows": 42}))
    with _patch_registry():
        results = await execute_graph(graph, http_client=client)

    statuses = {r.task_id: r.status for r in results}
    assert statuses["t1"] == TaskStatus.COMPLETED
    assert statuses["t2"] == TaskStatus.COMPLETED
    assert statuses["t3"] == TaskStatus.COMPLETED


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Timeout → FAILED; dependents → SKIPPED
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_timeout_propagates_skip():
    timeout_client = httpx.AsyncClient(transport=_timeout_transport())

    graph = _graph(
        _node("t1", AgentName.SQL),
        _node("t2", AgentName.REPORT, depends_on=["t1"]),
    )
    with _patch_registry():
        results = await execute_graph(graph, http_client=timeout_client)

    statuses = {r.task_id: r.status for r in results}
    assert statuses["t1"] == TaskStatus.FAILED
    assert statuses["t2"] == TaskStatus.SKIPPED
    # Error message mentions timeout
    failed = next(r for r in results if r.task_id == "t1")
    assert "timed out" in (failed.error or "").lower()


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Parallel execution — tasks with no shared deps run concurrently
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_parallel_tasks_all_complete():
    """t1, t2, t3 have no dependencies — all should be dispatched in one gather."""
    graph = _graph(
        _node("t1", AgentName.SQL),
        _node("t2", AgentName.VIZ),
        _node("t3", AgentName.ML),
        _node("t4", AgentName.REPORT, depends_on=["t1", "t2", "t3"]),
    )
    client = httpx.AsyncClient(transport=_ok_transport())
    with _patch_registry():
        results = await execute_graph(graph, http_client=client)

    for r in results:
        assert r.status == TaskStatus.COMPLETED


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Dependency chain: sequential order respected
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_dependency_chain_order():
    """Verify that t2 starts only after t1 completes (by inspecting started_at)."""
    graph = _graph(
        _node("t1", AgentName.CONTEXT),
        _node("t2", AgentName.SQL, depends_on=["t1"]),
    )
    client = httpx.AsyncClient(transport=_ok_transport())
    with _patch_registry():
        results = await execute_graph(graph, http_client=client)

    by_id = {r.task_id: r for r in results}
    # t2 must start after t1 ended
    assert by_id["t2"].started_at >= by_id["t1"].ended_at  # type: ignore[operator]


# ═══════════════════════════════════════════════════════════════════════════════
# 5. Unknown agent → FAILED, does not crash the engine
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_unknown_agent_fails_gracefully():
    # Patch registry without "nlp"
    sparse_registry = {k: v for k, v in MOCK_REGISTRY.items() if k != "nlp"}
    mock_settings = MagicMock()
    mock_settings.agent_registry     = sparse_registry
    mock_settings.agent_timeout_seconds = 30.0

    graph = _graph(
        _node("t1", AgentName.NLP),   # NLP not in registry
        _node("t2", AgentName.REPORT, depends_on=["t1"]),
    )
    client = httpx.AsyncClient(transport=_ok_transport())
    with patch("app.executor.get_settings", return_value=mock_settings):
        results = await execute_graph(graph, http_client=client)

    statuses = {r.task_id: r.status for r in results}
    assert statuses["t1"] == TaskStatus.FAILED
    assert statuses["t2"] == TaskStatus.SKIPPED


# ═══════════════════════════════════════════════════════════════════════════════
# 6. Hooks fired exactly once per task
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_hooks_called_once_per_task():
    started: list[str] = []
    done:    list[str] = []

    async def on_start(r: TaskResult) -> None:
        started.append(r.task_id)

    async def on_done(r: TaskResult) -> None:
        done.append(r.task_id)

    graph = _graph(
        _node("t1", AgentName.SQL),
        _node("t2", AgentName.VIZ),
        _node("t3", AgentName.REPORT, depends_on=["t1", "t2"]),
    )
    client = httpx.AsyncClient(transport=_ok_transport())
    with _patch_registry():
        await execute_graph(
            graph,
            on_task_start=on_start,
            on_task_done=on_done,
            http_client=client,
        )

    assert sorted(started) == ["t1", "t2", "t3"]
    assert sorted(done)    == ["t1", "t2", "t3"]


# ═══════════════════════════════════════════════════════════════════════════════
# 7. HTTP 500 from agent → FAILED, does not crash engine
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_http_error_marks_task_failed():
    def _error_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="Internal Server Error")

    client = httpx.AsyncClient(transport=httpx.MockTransport(_error_handler))
    graph = _graph(_node("t1", AgentName.SQL))
    with _patch_registry():
        results = await execute_graph(graph, http_client=client)

    assert results[0].status == TaskStatus.FAILED
    assert "500" in (results[0].error or "")
