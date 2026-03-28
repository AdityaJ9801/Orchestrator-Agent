"""Tests for the parallel execution engine (app/executor.py)."""
from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import httpx
import pytest

from app.executor import execute_graph
from app.models import AgentName, TaskGraph, TaskNode, TaskResult, TaskStatus


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


def _ok_transport(result: Any = None) -> httpx.MockTransport:
    if result is None:
        result = {"ok": True}

    def _handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=result)

    return httpx.MockTransport(_handler)


def _error_transport(status_code: int, body: str = "") -> httpx.MockTransport:
    def _handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status_code, text=body)

    return httpx.MockTransport(_handler)


def _timeout_transport() -> httpx.MockTransport:
    def _handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("timed out", request=request)

    return httpx.MockTransport(_handler)


MOCK_REGISTRY = {
    "context": "http://context-agent:8001",
    "sql":     "http://sql-agent:8002",
    "viz":     "http://viz-agent:8003",
    "ml":      "http://ml-agent:8004",
    "nlp":     "http://nlp-agent:8005",
    "report":  "http://report-agent:8006",
}


def _patch_registry(registry: dict | None = None):
    mock_settings = MagicMock()
    mock_settings.agent_registry = registry or MOCK_REGISTRY
    mock_settings.agent_timeout_seconds = 30.0
    return patch("app.executor.get_settings", return_value=mock_settings)


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


@pytest.mark.asyncio
async def test_timeout_propagates_skip():
    graph = _graph(
        _node("t1", AgentName.SQL),
        _node("t2", AgentName.REPORT, depends_on=["t1"]),
    )
    client = httpx.AsyncClient(transport=_timeout_transport())
    with _patch_registry():
        results = await execute_graph(graph, http_client=client)

    statuses = {r.task_id: r.status for r in results}
    assert statuses["t1"] == TaskStatus.FAILED
    assert statuses["t2"] == TaskStatus.SKIPPED
    failed = next(r for r in results if r.task_id == "t1")
    assert "timed out" in (failed.error or "").lower()


@pytest.mark.asyncio
async def test_parallel_tasks_all_complete():
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


@pytest.mark.asyncio
async def test_dependency_chain_order():
    graph = _graph(
        _node("t1", AgentName.CONTEXT),
        _node("t2", AgentName.SQL, depends_on=["t1"]),
    )
    client = httpx.AsyncClient(transport=_ok_transport())
    with _patch_registry():
        results = await execute_graph(graph, http_client=client)

    by_id = {r.task_id: r for r in results}
    assert by_id["t2"].started_at >= by_id["t1"].ended_at  # type: ignore[operator]


@pytest.mark.asyncio
async def test_unknown_agent_fails_gracefully():
    sparse = {k: v for k, v in MOCK_REGISTRY.items() if k != "nlp"}
    graph = _graph(
        _node("t1", AgentName.NLP),
        _node("t2", AgentName.REPORT, depends_on=["t1"]),
    )
    client = httpx.AsyncClient(transport=_ok_transport())
    with _patch_registry(sparse):
        results = await execute_graph(graph, http_client=client)

    statuses = {r.task_id: r.status for r in results}
    assert statuses["t1"] == TaskStatus.FAILED
    assert statuses["t2"] == TaskStatus.SKIPPED


@pytest.mark.asyncio
async def test_hooks_called_once_per_task():
    started: list[str] = []
    done: list[str] = []

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


@pytest.mark.asyncio
async def test_http_500_marks_task_failed():
    client = httpx.AsyncClient(transport=_error_transport(500, "Internal Server Error"))
    graph = _graph(_node("t1", AgentName.SQL))
    with _patch_registry():
        results = await execute_graph(graph, http_client=client)

    assert results[0].status == TaskStatus.FAILED
    assert "500" in (results[0].error or "")


@pytest.mark.asyncio
async def test_http_404_marks_task_failed():
    client = httpx.AsyncClient(transport=_error_transport(404, "Not Found"))
    graph = _graph(_node("t1", AgentName.SQL))
    with _patch_registry():
        results = await execute_graph(graph, http_client=client)

    assert results[0].status == TaskStatus.FAILED
    assert "404" in (results[0].error or "")


@pytest.mark.asyncio
async def test_dependency_context_forwarded():
    """Downstream task payload must include _context key."""
    received_payloads: list[dict] = []

    def _capture(request: httpx.Request) -> httpx.Response:
        import json as _json
        try:
            body = _json.loads(request.content)
            received_payloads.append(body)
        except Exception:
            pass
        return httpx.Response(200, json={"ok": True})

    graph = _graph(
        _node("t1", AgentName.CONTEXT),
        _node("t2", AgentName.SQL, depends_on=["t1"]),
    )
    client = httpx.AsyncClient(transport=httpx.MockTransport(_capture))
    with _patch_registry():
        await execute_graph(graph, http_client=client)

    # t2 must have received a payload with _context key
    t2_payload = next((p for p in received_payloads if "_context" in p), None)
    assert t2_payload is not None, "t2 payload should contain _context key"
    # _context is a dict (may be empty if t1 result was None, but key must exist)
    assert isinstance(t2_payload["_context"], dict)


@pytest.mark.asyncio
async def test_empty_graph_returns_empty_list():
    graph = TaskGraph(intent="empty", tasks=[])
    with _patch_registry():
        results = await execute_graph(graph)
    assert results == []


@pytest.mark.asyncio
async def test_single_task_no_deps_completes():
    graph = _graph(_node("t1", AgentName.REPORT))
    client = httpx.AsyncClient(transport=_ok_transport({"report": "done"}))
    with _patch_registry():
        results = await execute_graph(graph, http_client=client)

    assert len(results) == 1
    assert results[0].status == TaskStatus.COMPLETED
    assert results[0].result == {"report": "done"}


@pytest.mark.asyncio
async def test_all_tasks_fail_returns_results():
    graph = _graph(
        _node("t1", AgentName.SQL),
        _node("t2", AgentName.VIZ),
    )
    client = httpx.AsyncClient(transport=_error_transport(503, "Service Unavailable"))
    with _patch_registry():
        results = await execute_graph(graph, http_client=client)

    assert len(results) == 2
    for r in results:
        assert r.status == TaskStatus.FAILED


@pytest.mark.asyncio
async def test_task_result_duration_ms():
    graph = _graph(_node("t1", AgentName.SQL))
    client = httpx.AsyncClient(transport=_ok_transport())
    with _patch_registry():
        results = await execute_graph(graph, http_client=client)

    assert results[0].duration_ms is not None
    assert results[0].duration_ms >= 0
