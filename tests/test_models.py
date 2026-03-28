"""
Tests for Pydantic models (app/models.py).

Scenarios:
  1.  TaskNode default task_id is generated
  2.  TaskGraph validates correctly
  3.  TaskResult duration_ms computed correctly
  4.  TaskResult duration_ms is None when timestamps missing
  5.  AnalyzeRequest validates min_length on query
  6.  AgentName enum values are correct strings
  7.  TaskStatus enum values are correct strings
  8.  SSEEvent model validates
  9.  AgentsStatusResponse model validates
  10. AnalyzeResponse partial flag defaults to False
"""
from __future__ import annotations

from datetime import datetime, timezone, timedelta

import pytest
from pydantic import ValidationError

from app.models import (
    AgentName,
    AgentsStatusResponse,
    AnalyzeRequest,
    AnalyzeResponse,
    SSEEvent,
    TaskGraph,
    TaskNode,
    TaskResult,
    TaskStatus,
)


def test_task_node_auto_task_id():
    node = TaskNode(agent=AgentName.SQL, description="test")
    assert node.task_id
    assert len(node.task_id) > 0


def test_task_node_custom_task_id():
    node = TaskNode(task_id="custom-id", agent=AgentName.SQL, description="test")
    assert node.task_id == "custom-id"


def test_task_graph_valid():
    graph = TaskGraph(
        intent="test intent",
        tasks=[TaskNode(agent=AgentName.SQL, description="query")],
    )
    assert graph.intent == "test intent"
    assert len(graph.tasks) == 1


def test_task_graph_empty_tasks():
    graph = TaskGraph(intent="empty", tasks=[])
    assert graph.tasks == []


def test_task_result_duration_ms():
    now = datetime.now(tz=timezone.utc)
    later = now + timedelta(milliseconds=250)
    result = TaskResult(
        task_id="t1",
        agent=AgentName.SQL,
        started_at=now,
        ended_at=later,
    )
    assert result.duration_ms is not None
    assert abs(result.duration_ms - 250.0) < 1.0


def test_task_result_duration_ms_none_when_no_timestamps():
    result = TaskResult(task_id="t1", agent=AgentName.SQL)
    assert result.duration_ms is None


def test_task_result_default_status_pending():
    result = TaskResult(task_id="t1", agent=AgentName.SQL)
    assert result.status == TaskStatus.PENDING


def test_analyze_request_valid():
    req = AnalyzeRequest(query="Show me revenue data", context_id="ctx-123")
    assert req.query == "Show me revenue data"
    assert req.session_id is None


def test_analyze_request_min_length_violation():
    with pytest.raises(ValidationError):
        AnalyzeRequest(query="Hi", context_id="ctx-123")  # min_length=3, "Hi" is 2


def test_analyze_request_with_session_id():
    req = AnalyzeRequest(query="Show data", context_id="ctx-1", session_id="sess-abc")
    assert req.session_id == "sess-abc"


def test_agent_name_enum_values():
    assert AgentName.CONTEXT.value == "context"
    assert AgentName.SQL.value     == "sql"
    assert AgentName.VIZ.value     == "viz"
    assert AgentName.ML.value      == "ml"
    assert AgentName.NLP.value     == "nlp"
    assert AgentName.REPORT.value  == "report"


def test_task_status_enum_values():
    assert TaskStatus.PENDING.value   == "pending"
    assert TaskStatus.RUNNING.value   == "running"
    assert TaskStatus.COMPLETED.value == "completed"
    assert TaskStatus.FAILED.value    == "failed"
    assert TaskStatus.SKIPPED.value   == "skipped"


def test_sse_event_model():
    event = SSEEvent(event="task_start", data={"task_id": "t1", "agent": "sql"})
    assert event.event == "task_start"
    assert event.data["agent"] == "sql"


def test_agents_status_response():
    from app.models import AgentHealthInfo
    resp = AgentsStatusResponse(agents=[
        AgentHealthInfo(agent="sql", url="http://sql:8002", healthy=True, latency_ms=12.5),
    ])
    assert len(resp.agents) == 1
    assert resp.agents[0].healthy is True


def test_analyze_response_partial_defaults_false():
    graph = TaskGraph(intent="test", tasks=[])
    resp = AnalyzeResponse(
        task_id="task-1",
        session_id="sess-1",
        intent="test",
        graph=graph,
        results=[],
    )
    assert resp.partial is False
