"""
LLM Planning layer -- converts a natural-language query into a TaskGraph.

Provider switching:
  * LLM_PROVIDER=stub   --> returns a canned task graph (no LLM needed, for local testing)
  * LLM_PROVIDER=ollama --> calls Ollama HTTP API (free / local)
  * LLM_PROVIDER=claude --> calls Anthropic SDK (paid)

Retry logic:
  On first parse failure, a correction prompt is sent once more.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict

import httpx
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_fixed,
)

from app.config import get_settings
from app.models import AgentName, TaskGraph, TaskNode

logger = logging.getLogger(__name__)

# â”€â”€ Prompt templates â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

_SYSTEM_PROMPT = """You are a data-analysis orchestration planner.
Given a user query and a list of available specialist agents, produce a JSON task graph.

Available agents: context, sql, viz, ml, nlp, report

Output ONLY valid JSON matching this exact schema (no markdown, no prose):
{
  "intent": "<one-sentence intent summary>",
  "tasks": [
    {
      "task_id": "<short alphanumeric id>",
      "agent": "<one of: context|sql|viz|ml|nlp|report>",
      "description": "<what this agent should do>",
      "payload": { "<key>": "<value>" },
      "depends_on": ["<task_id>", ...]
    }
  ]
}

Rules:
- Tasks with no dependencies run in parallel.
- Later tasks list their prerequisite task_ids in depends_on.
- Always include a 'report' task as the final aggregator.
- Keep payload concise â€” agent-specific hints only.
"""

_USER_TEMPLATE = "Query: {query}\nContext ID: {context_id}"

_CORRECTION_TEMPLATE = """Your previous response was not valid JSON. Here is the error:
{error}

Please respond ONLY with valid JSON matching the schema. No markdown. No explanation."""


# â”€â”€ Internal helpers â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def _extract_json(text: str) -> str:
    """Strip markdown fences and return the first JSON object found."""
    # Remove ```json ... ``` fences
    text = re.sub(r"```(?:json)?\s*", "", text).strip()
    text = text.rstrip("`").strip()
    # Find first { ... } block
    start = text.find("{")
    end   = text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("No JSON object found in LLM response")
    return text[start : end + 1]


def _parse_graph(raw: str) -> TaskGraph:
    """Parse raw LLM output into a validated TaskGraph."""
    cleaned = _extract_json(raw)
    data    = json.loads(cleaned)

    # Normalise agent names to enum
    for t in data.get("tasks", []):
        t["agent"] = t["agent"].lower().strip()

    return TaskGraph(**data)



async def _call_stub(query: str) -> str:
    """
    Return a canned TaskGraph JSON without calling any LLM.
    Performs simple keyword matching to pick a realistic agent mix.
    Used when LLM_PROVIDER=stub for local integration testing.
    """
    q = query.lower()
    tasks: list[dict] = []

    # Always start with context
    tasks.append({
        "task_id":     "t01",
        "agent":       "context",
        "description": "Fetch dataset schema and metadata",
        "payload":     {"query": query},
        "depends_on":  [],
    })

    # SQL for data retrieval queries
    if any(w in q for w in ["revenue", "sales", "total", "region", "sum", "count",
                             "group", "last", "quarter", "month", "list", "show",
                             "compare", "data", "statistics", "schema"]):
        tasks.append({
            "task_id":     "t02",
            "agent":       "sql",
            "description": "Query database for relevant metrics",
            "payload":     {"query": query},
            "depends_on":  ["t01"],
        })

    # ML for prediction / forecast / model queries
    if any(w in q for w in ["predict", "forecast", "model", "ml", "regression",
                             "feature", "xgb", "train", "importance", "score"]):
        dep = "t02" if any(t["task_id"] == "t02" for t in tasks) else "t01"
        tasks.append({
            "task_id":     "t03",
            "agent":       "ml",
            "description": "Train model and generate predictions",
            "payload":     {"query": query},
            "depends_on":  [dep],
        })

    # NLP for sentiment / text / feedback queries
    if any(w in q for w in ["sentiment", "nlp", "feedback", "topic", "summary",
                             "text", "review", "opinion", "analyse feedback"]):
        dep = "t02" if any(t["task_id"] == "t02" for t in tasks) else "t01"
        tasks.append({
            "task_id":     "t04",
            "agent":       "nlp",
            "description": "Run NLP analysis on text data",
            "payload":     {"query": query},
            "depends_on":  [dep],
        })

    # VIZ for chart / visualisation queries
    if any(w in q for w in ["chart", "plot", "graph", "visual", "bar", "line",
                             "trend", "visualis"]):
        dep = "t02" if any(t["task_id"] == "t02" for t in tasks) else "t01"
        tasks.append({
            "task_id":     "t05",
            "agent":       "viz",
            "description": "Generate visualisations",
            "payload":     {"query": query},
            "depends_on":  [dep],
        })

    # Always end with report
    last_deps = [t["task_id"] for t in tasks if t["task_id"] != "t01"]
    if not last_deps:
        last_deps = ["t01"]
    tasks.append({
        "task_id":     "t99",
        "agent":       "report",
        "description": "Aggregate all results into a final report",
        "payload":     {},
        "depends_on":  last_deps,
    })

    # Build a one-sentence intent
    intent = f"Analyse query: '{query[:80]}'"
    graph_dict = {"intent": intent, "tasks": tasks}
    return json.dumps(graph_dict)


async def _llm_call(messages: list[dict], settings) -> str:
    provider = settings.llm_provider

    if provider == "stub":
        user_content = next(
            (m["content"] for m in messages if m["role"] == "user"), ""
        )
        import re as _re
        m = _re.search(r"Query: (.+?)(?:\nContext ID|\.?\.?\.?$)", user_content, _re.DOTALL)
        query = m.group(1).strip() if m else user_content
        return await _call_stub(query)

    from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
    from langchain_groq import ChatGroq
    from langchain_openai import ChatOpenAI
    from langchain_anthropic import ChatAnthropic
    from langchain_community.chat_models import ChatOllama

    # Auto-select Groq when API key is present and provider isn't forced to ollama
    if getattr(settings, "XAI_API_KEY", "") and provider not in ("ollama", "claude", "stub"):
        logger.info("Orchestrator LLM: Grok (XAI_API_KEY detected)")
        provider = "grok"
    elif getattr(settings, "groq_api_key", "") and provider not in ("ollama", "claude", "stub"):
        logger.info("Orchestrator LLM: Groq (GROQ_API_KEY detected)")
        provider = "groq"

    lc_messages = []
    for m in messages:
        if m["role"] == "system":
            lc_messages.append(SystemMessage(content=m["content"]))
        elif m["role"] == "user":
            lc_messages.append(HumanMessage(content=m["content"]))
        elif m["role"] == "assistant":
            lc_messages.append(AIMessage(content=m["content"]))

    if provider == "claude":
        llm = ChatAnthropic(api_key=settings.anthropic_api_key, model_name=settings.claude_model, temperature=settings.planning_temperature)
    elif provider == "groq":
        llm = ChatGroq(api_key=settings.groq_api_key, model_name=settings.groq_model, temperature=settings.planning_temperature)
    elif provider == "grok":
        llm = ChatOpenAI(api_key=settings.XAI_API_KEY, base_url="https://api.x.ai/v1", model="grok-2-latest", temperature=settings.planning_temperature)
    elif provider == "mcp":
        # MCP Placeholder
        llm = ChatOpenAI(api_key=getattr(settings, "OPENAI_API_KEY", ""), model_name="gpt-4o", temperature=settings.planning_temperature)
    else:
        llm = ChatOllama(base_url=settings.ollama_base_url, model=settings.ollama_model, temperature=settings.planning_temperature)

    response = await llm.ainvoke(lc_messages)
    return response.content


# â”€â”€ Public planning function â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

async def plan_query(query: str, context_id: str) -> TaskGraph:
    """
    Ask the configured LLM to decompose the query into a TaskGraph.
    Retries once with a correction prompt on JSON parse errors.
    """
    settings = get_settings()

    messages: list[dict[str, str]] = [
        {"role": "system",    "content": _SYSTEM_PROMPT},
        {"role": "user",      "content": _USER_TEMPLATE.format(
            query=query, context_id=context_id
        )},
    ]

    raw = ""
    for attempt in range(settings.planning_max_retries + 1):
        try:
            raw = await _llm_call(messages, settings)
            logger.debug("LLM raw output (attempt %d): %s", attempt, raw[:300])
            graph = _parse_graph(raw)
            logger.info("Plan produced: %d tasks, intent=%r", len(graph.tasks), graph.intent)
            return graph
        except (json.JSONDecodeError, KeyError, ValueError) as exc:
            if attempt < settings.planning_max_retries:
                logger.warning("JSON parse error (attempt %d): %s â€” retrying with correction", attempt, exc)
                messages.append({"role": "assistant", "content": raw})
                messages.append({
                    "role": "user",
                    "content": _CORRECTION_TEMPLATE.format(error=str(exc)),
                })
            else:
                logger.error("LLM planning failed after %d attempts: %s", attempt + 1, exc)
                raise RuntimeError(
                    f"Orchestrator could not produce a valid task graph after "
                    f"{settings.planning_max_retries + 1} attempt(s). Last error: {exc}"
                ) from exc

    # Should never reach here
    raise RuntimeError("Unexpected exit from planning loop")
