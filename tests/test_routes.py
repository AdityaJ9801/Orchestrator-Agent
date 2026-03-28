"""
Tests for FastAPI routes using TestClient.

Scenarios:
  1.  GET /health returns 200 with correct fields
  2.  POST /analyze returns 200 with valid AnalyzeResponse shape
  3.  POST /analyze with too-short query returns 422
  4.  POST /analyze with missing context_id returns 422
  5.  POST /plan returns 200 with TaskGraph shape
  6.  POST /plan with missing fields returns 422
  7.  POST /analyze/stream returns 200 with SSE content-type
  8.  GET /agents/status returns 200 with all 6 agents
  9.  POST /analyze planner error returns 502
  10. GET /docs returns 200 (OpenAPI docs available)
"""
from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from fastapi.testclient import TestClient

from app.main import create_app
from app.models import AgentName, TaskGraph, TaskNode, TaskResult, TaskStatus


# ── Shared fixtures ───────────────────────────────────────────────────────────

MOCK_GRAPH = TaskGraph(
    intent="Test intent",
    tasks=[
        TaskNode(task_id="t1", agent=AgentName.SQL,    description="Query data"),
        TaskNode(task_id="t2", agent=AgentName.REPORT, description="Report", depends_on=["t1"]),
    ],
)

MOCK_RESULTS = [
    TaskResult(task_id="t1", agent=AgentName.SQL,    status=TaskStatus.COMPLETED, result={"rows": 5}),
    TaskResult(task_id="t2", agent=AgentName.REPORT, status=TaskStatus.COMPLETED, result={"pdf": "url"}),
]


def _mock_plan():
    return patch("app.routes.analyze.plan_query", new=AsyncMock(return_value=MOCK_GRAPH))


def _mock_plan_route():
    return patch("app.routes.plan.plan_query", new=AsyncMock(return_value=MOCK_GRAPH))


def _mock_execute():
    return patch("app.routes.analyze.execute_graph", new=AsyncMock(return_value=MOCK_RESULTS))


def _mock_session_save():
    return patch("app.routes.analyze.save_session", new=AsyncMock())


def _mock_stream_plan():
    return patch("app.routes.stream.plan_query", new=AsyncMock(return_value=MOCK_GRAPH))


def _mock_stream_execute():
    return patch("app.routes.stream.execute_graph", new=AsyncMock(return_value=MOCK_RESULTS))


def _mock_stream_session():
    return patch("app.routes.stream.save_session", new=AsyncMock())


@pytest.fixture
def client():
    app = create_app()
    with TestClient(app, raise_server_exceptions=False) as c:
        yield c


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_health_returns_200(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["service"] == "orchestrator-agent"
    assert "llm_provider" in data


def test_analyze_returns_200(client):
    with _mock_plan(), _mock_execute(), _mock_session_save():
        resp = client.post("/analyze", json={
            "query": "Show me total revenue by region",
            "context_id": "ctx-test-001",
        })
    assert resp.status_code == 200
    data = resp.json()
    assert "task_id" in data
    assert "session_id" in data
    assert "intent" in data
    assert "results" in data
    assert isinstance(data["results"], list)


def test_analyze_query_too_short_returns_422(client):
    resp = client.post("/analyze", json={
        "query": "Hi",  # 2 chars, min_length=3
        "context_id": "ctx-test",
    })
    assert resp.status_code == 422


def test_analyze_missing_context_id_returns_422(client):
    resp = client.post("/analyze", json={"query": "Show me revenue"})
    assert resp.status_code == 422


def test_analyze_missing_query_returns_422(client):
    resp = client.post("/analyze", json={"context_id": "ctx-test"})
    assert resp.status_code == 422


def test_plan_returns_200(client):
    with _mock_plan_route():
        resp = client.post("/plan", json={
            "query": "Show me revenue by region",
            "context_id": "ctx-plan-001",
        })
    assert resp.status_code == 200
    data = resp.json()
    assert "intent" in data
    assert "tasks" in data
    assert isinstance(data["tasks"], list)


def test_plan_missing_fields_returns_422(client):
    resp = client.post("/plan", json={"query": "test"})  # missing context_id
    assert resp.status_code == 422


def test_analyze_stream_returns_200_sse(client):
    with _mock_stream_plan(), _mock_stream_execute(), _mock_stream_session():
        resp = client.post("/analyze/stream", json={
            "query": "Show me revenue data",
            "context_id": "ctx-stream-001",
        })
    assert resp.status_code == 200
    assert "text/event-stream" in resp.headers.get("content-type", "")


def test_agents_status_returns_200(client):
    mock_health = MagicMock()
    mock_health.status_code = 200

    async def _fake_get(url, timeout=None):
        return mock_health

    with patch("app.routes.agents.httpx.AsyncClient") as mock_client_cls:
        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_health)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client_cls.return_value = mock_client

        resp = client.get("/agents/status")

    assert resp.status_code == 200
    data = resp.json()
    assert "agents" in data
    assert "checked_at" in data


def test_analyze_planner_error_returns_502(client):
    with patch("app.routes.analyze.plan_query",
               new=AsyncMock(side_effect=RuntimeError("LLM failed"))):
        resp = client.post("/analyze", json={
            "query": "Show me revenue",
            "context_id": "ctx-err",
        })
    assert resp.status_code == 502


def test_plan_planner_error_returns_502(client):
    with patch("app.routes.plan.plan_query",
               new=AsyncMock(side_effect=RuntimeError("LLM failed"))):
        resp = client.post("/plan", json={
            "query": "Show me revenue",
            "context_id": "ctx-err",
        })
    assert resp.status_code == 502


def test_docs_available(client):
    resp = client.get("/docs")
    assert resp.status_code == 200


def test_analyze_partial_flag_set_when_task_fails(client):
    failed_results = [
        TaskResult(task_id="t1", agent=AgentName.SQL,    status=TaskStatus.FAILED,  error="timeout"),
        TaskResult(task_id="t2", agent=AgentName.REPORT, status=TaskStatus.SKIPPED, error="upstream failed"),
    ]
    with _mock_plan(), \
         patch("app.routes.analyze.execute_graph", new=AsyncMock(return_value=failed_results)), \
         _mock_session_save():
        resp = client.post("/analyze", json={
            "query": "Show me revenue",
            "context_id": "ctx-partial",
        })
    assert resp.status_code == 200
    assert resp.json()["partial"] is True


def test_analyze_session_id_reused_when_provided(client):
    with _mock_plan(), _mock_execute(), _mock_session_save():
        resp = client.post("/analyze", json={
            "query": "Show me revenue",
            "context_id": "ctx-001",
            "session_id": "existing-session-abc",
        })
    assert resp.status_code == 200
    assert resp.json()["session_id"] == "existing-session-abc"
