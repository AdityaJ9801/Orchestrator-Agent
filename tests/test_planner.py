"""
Unit tests for the intent planner — 10 sample queries.

These tests mock the LLM provider so no network calls are made.
They verify:
  1. Correct JSON is parsed into a valid TaskGraph.
  2. The correction-prompt retry fires on malformed JSON.
  3. All 10 sample queries produce non-empty task graphs.
"""
from __future__ import annotations

import json
from unittest.mock import AsyncMock, patch

import pytest

from app.models import AgentName, TaskGraph
from app.planner import _extract_json, _parse_graph, plan_query

# ── Golden LLM response template ────────────────────────────────────────────

def _make_llm_response(intent: str, tasks: list[dict]) -> str:
    return json.dumps({"intent": intent, "tasks": tasks})


def _simple_plan(intent: str = "Analyse revenue trends") -> str:
    return _make_llm_response(
        intent,
        [
            {
                "task_id": "t1",
                "agent": "context",
                "description": "Fetch context",
                "payload": {},
                "depends_on": [],
            },
            {
                "task_id": "t2",
                "agent": "sql",
                "description": "Query revenue data",
                "payload": {"table": "sales"},
                "depends_on": ["t1"],
            },
            {
                "task_id": "t3",
                "agent": "report",
                "description": "Assemble report",
                "payload": {},
                "depends_on": ["t2"],
            },
        ],
    )


# ── Helper: patch the LLM call and call plan_query ──────────────────────────

async def _plan_with_mock(query: str, response_text: str) -> TaskGraph:
    """Run plan_query with LLM mocked to return *response_text*."""
    with patch("app.planner._llm_call", new=AsyncMock(return_value=response_text)):
        return await plan_query(query, context_id="ctx-test-123")


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Basic extraction helpers
# ═══════════════════════════════════════════════════════════════════════════════

def test_extract_json_plain():
    raw = '{"intent":"ok","tasks":[]}'
    assert _extract_json(raw) == raw


def test_extract_json_strips_fences():
    raw = '```json\n{"intent":"ok","tasks":[]}\n```'
    assert _extract_json(raw) == '{"intent":"ok","tasks":[]}'


def test_extract_json_strips_triple_backtick_no_lang():
    raw = '```\n{"intent":"x","tasks":[]}\n```'
    result = _extract_json(raw)
    assert result.startswith("{")


def test_parse_graph_valid():
    raw = _simple_plan("test")
    graph = _parse_graph(raw)
    assert isinstance(graph, TaskGraph)
    assert len(graph.tasks) == 3
    assert graph.tasks[-1].agent == AgentName.REPORT


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Ten sample queries — each should yield a non-empty TaskGraph
# ═══════════════════════════════════════════════════════════════════════════════

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
    mock_response = _simple_plan(intent=f"Analysis: {query[:40]}")
    graph = await _plan_with_mock(query, mock_response)

    assert isinstance(graph, TaskGraph)
    assert len(graph.tasks) > 0
    assert graph.intent  # non-empty intent string


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Retry / correction-prompt mechanism
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_retry_fires_on_bad_json():
    """
    First call returns garbage → planner should retry once with correction prompt.
    Second call returns valid JSON → result must be a TaskGraph.
    """
    good_response = _simple_plan("Recovery test")

    call_count = 0

    async def _fake_llm(messages, settings):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return "This is not JSON at all 🙃"
        return good_response

    with patch("app.planner._llm_call", new=_fake_llm):
        graph = await plan_query("Show sales data", context_id="ctx-retry")

    assert call_count == 2, "LLM should have been called twice (first fail + retry)"
    assert isinstance(graph, TaskGraph)


@pytest.mark.asyncio
async def test_retry_exhausted_raises():
    """If both attempts return bad JSON, RuntimeError must be raised."""

    async def _always_bad(messages, settings):
        return "definitely-not-json"

    with patch("app.planner._llm_call", new=_always_bad):
        with pytest.raises(RuntimeError, match="valid task graph"):
            await plan_query("Anything", context_id="ctx-fail")


@pytest.mark.asyncio
async def test_markdown_fenced_response_parsed_correctly():
    """LLM wraps JSON in markdown fences — planner must still parse it."""
    raw = f"```json\n{_simple_plan('Fenced test')}\n```"
    graph = await _plan_with_mock("What are top SKUs?", raw)
    assert isinstance(graph, TaskGraph)


@pytest.mark.asyncio
async def test_agent_names_normalised_to_lowercase():
    """LLM returns upper-case agent names — they must be normalised."""
    payload = {
        "intent": "Normalise test",
        "tasks": [
            {
                "task_id": "n1",
                "agent": "SQL",       # uppercase
                "description": "Query",
                "payload": {},
                "depends_on": [],
            },
            {
                "task_id": "n2",
                "agent": "Report",    # mixed-case
                "description": "Report",
                "payload": {},
                "depends_on": ["n1"],
            },
        ],
    }
    graph = await _plan_with_mock("Test query", json.dumps(payload))
    agents = [t.agent for t in graph.tasks]
    assert AgentName.SQL in agents
    assert AgentName.REPORT in agents
