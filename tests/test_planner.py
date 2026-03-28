"""
Tests for the LLM planning layer (app/planner.py).

Scenarios:
  1.  JSON extraction helpers work correctly
  2.  Markdown-fenced responses are parsed
  3.  Agent names normalised to lowercase
  4.  10 sample queries produce valid TaskGraphs
  5.  Retry fires on bad JSON, succeeds on second attempt
  6.  Both retries exhausted → RuntimeError raised
  7.  Stub provider returns correct agents for keyword queries
  8.  Stub always ends with report task
"""
from __future__ import annotations

import json
from unittest.mock import AsyncMock, patch

import pytest

from app.models import AgentName, TaskGraph
from app.planner import _extract_json, _parse_graph, _call_stub, plan_query


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_plan(intent: str = "Test intent", tasks: list[dict] | None = None) -> str:
    if tasks is None:
        tasks = [
            {"task_id": "t1", "agent": "context", "description": "Fetch context",
             "payload": {}, "depends_on": []},
            {"task_id": "t2", "agent": "sql",     "description": "Query data",
             "payload": {"table": "sales"}, "depends_on": ["t1"]},
            {"task_id": "t3", "agent": "report",  "description": "Assemble report",
             "payload": {}, "depends_on": ["t2"]},
        ]
    return json.dumps({"intent": intent, "tasks": tasks})


async def _plan_with_mock(query: str, response_text: str) -> TaskGraph:
    with patch("app.planner._llm_call", new=AsyncMock(return_value=response_text)):
        return await plan_query(query, context_id="ctx-test-123")


# ── JSON extraction ───────────────────────────────────────────────────────────

def test_extract_json_plain():
    raw = '{"intent":"ok","tasks":[]}'
    assert _extract_json(raw) == raw


def test_extract_json_strips_json_fence():
    raw = '```json\n{"intent":"ok","tasks":[]}\n```'
    assert _extract_json(raw) == '{"intent":"ok","tasks":[]}'


def test_extract_json_strips_plain_fence():
    raw = '```\n{"intent":"x","tasks":[]}\n```'
    result = _extract_json(raw)
    assert result.startswith("{")
    assert result.endswith("}")


def test_extract_json_raises_on_no_braces():
    with pytest.raises(ValueError, match="No JSON object found"):
        _extract_json("this has no json at all")


def test_extract_json_with_surrounding_prose():
    raw = 'Here is the plan: {"intent":"ok","tasks":[]} Hope that helps!'
    result = _extract_json(raw)
    assert result == '{"intent":"ok","tasks":[]}'


# ── Graph parsing ─────────────────────────────────────────────────────────────

def test_parse_graph_valid():
    graph = _parse_graph(_make_plan("test"))
    assert isinstance(graph, TaskGraph)
    assert len(graph.tasks) == 3
    assert graph.tasks[-1].agent == AgentName.REPORT


def test_parse_graph_normalises_uppercase_agents():
    payload = json.dumps({
        "intent": "Normalise test",
        "tasks": [
            {"task_id": "n1", "agent": "SQL",    "description": "Query",  "payload": {}, "depends_on": []},
            {"task_id": "n2", "agent": "Report", "description": "Report", "payload": {}, "depends_on": ["n1"]},
        ],
    })
    graph = _parse_graph(payload)
    agents = [t.agent for t in graph.tasks]
    assert AgentName.SQL    in agents
    assert AgentName.REPORT in agents


def test_parse_graph_invalid_json_raises():
    with pytest.raises((json.JSONDecodeError, ValueError)):
        _parse_graph("not json at all")


# ── Stub provider ─────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_stub_always_has_report():
    raw = await _call_stub("show me revenue data")
    data = json.loads(raw)
    agents = [t["agent"] for t in data["tasks"]]
    assert "report" in agents
    assert data["tasks"][-1]["agent"] == "report"


@pytest.mark.asyncio
async def test_stub_sql_for_revenue_query():
    raw = await _call_stub("show total revenue by region")
    data = json.loads(raw)
    agents = [t["agent"] for t in data["tasks"]]
    assert "sql" in agents


@pytest.mark.asyncio
async def test_stub_ml_for_predict_query():
    raw = await _call_stub("predict next quarter revenue")
    data = json.loads(raw)
    agents = [t["agent"] for t in data["tasks"]]
    assert "ml" in agents


@pytest.mark.asyncio
async def test_stub_nlp_for_sentiment_query():
    raw = await _call_stub("analyse customer feedback sentiment")
    data = json.loads(raw)
    agents = [t["agent"] for t in data["tasks"]]
    assert "nlp" in agents


@pytest.mark.asyncio
async def test_stub_viz_for_chart_query():
    raw = await _call_stub("show a bar chart of sales")
    data = json.loads(raw)
    agents = [t["agent"] for t in data["tasks"]]
    assert "viz" in agents


@pytest.mark.asyncio
async def test_stub_context_always_first():
    raw = await _call_stub("any query at all")
    data = json.loads(raw)
    assert data["tasks"][0]["agent"] == "context"


# ── 10 sample queries ─────────────────────────────────────────────────────────

SAMPLE_QUERIES = [
    "What were total sales by region last quarter?",
    "Show me a bar chart of monthly revenue for 2024.",
    "Predict next quarter's churn rate using historical data.",
    "Which customers have not placed an order in 90 days?",
    "Summarise the sentiment of product reviews from January.",
    "Compare ARPU across subscription tiers.",
    "Run a cohort analysis on users who signed up in Q1.",
    "Generate a PDF report of key metrics for the board.",
    "Detect anomalies in daily transaction volumes.",
    "What is the correlation between marketing spend and new sign-ups?",
]


@pytest.mark.asyncio
@pytest.mark.parametrize("query", SAMPLE_QUERIES)
async def test_plan_query_sample(query: str):
    """Each sample query with a mocked LLM returns a valid TaskGraph."""
    mock_response = _make_plan(intent=f"Analysis: {query[:40]}")
    graph = await _plan_with_mock(query, mock_response)
    assert isinstance(graph, TaskGraph)
    assert len(graph.tasks) > 0
    assert graph.intent


# ── Retry / correction-prompt ─────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_retry_fires_on_bad_json():
    """First call returns garbage → retry once → second call returns valid JSON."""
    good_response = _make_plan("Recovery test")
    call_count = 0

    async def _fake_llm(messages, settings):
        nonlocal call_count
        call_count += 1
        return "not json" if call_count == 1 else good_response

    with patch("app.planner._llm_call", new=_fake_llm):
        graph = await plan_query("Show sales data", context_id="ctx-retry")

    assert call_count == 2
    assert isinstance(graph, TaskGraph)


@pytest.mark.asyncio
async def test_retry_exhausted_raises():
    """Both attempts return bad JSON → RuntimeError."""
    async def _always_bad(messages, settings):
        return "definitely-not-json"

    with patch("app.planner._llm_call", new=_always_bad):
        with pytest.raises(RuntimeError, match="valid task graph"):
            await plan_query("Anything", context_id="ctx-fail")


@pytest.mark.asyncio
async def test_markdown_fenced_response_parsed():
    """LLM wraps JSON in markdown fences — planner must still parse it."""
    raw = f"```json\n{_make_plan('Fenced test')}\n```"
    graph = await _plan_with_mock("What are top SKUs?", raw)
    assert isinstance(graph, TaskGraph)


@pytest.mark.asyncio
async def test_correction_prompt_appended_on_retry():
    """On retry, the correction prompt must be appended to messages."""
    messages_seen: list[list[dict]] = []
    call_count = 0

    async def _fake_llm(messages, settings):
        nonlocal call_count
        call_count += 1
        messages_seen.append(list(messages))
        return "bad json" if call_count == 1 else _make_plan("Corrected")

    with patch("app.planner._llm_call", new=_fake_llm):
        await plan_query("test", context_id="ctx-correction")

    # Second call should have more messages (original + assistant + correction)
    assert len(messages_seen[1]) > len(messages_seen[0])
    roles_second = [m["role"] for m in messages_seen[1]]
    assert "assistant" in roles_second
