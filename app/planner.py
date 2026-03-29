"""
LLM Planning layer -- converts a natural-language query into a TaskGraph.

Provider switching:
  * LLM_PROVIDER=stub      --> canned task graph (no LLM, for local testing)
  * LLM_PROVIDER=ollama    --> Ollama HTTP API (free / local)
  * LLM_PROVIDER=groq      --> Groq API
  * LLM_PROVIDER=grok      --> xAI Grok API
  * LLM_PROVIDER=anthropic --> Anthropic Claude API

Retry logic:
  On first parse failure, a correction prompt is sent once more.
"""
from __future__ import annotations

import json
import logging
import re

from app.config import get_settings
from app.models import TaskGraph

logger = logging.getLogger(__name__)

# ── Prompt templates ──────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """You are a data-analysis orchestration planner.
Given a user query and a context_id (cached dataset schema), produce a JSON task graph.

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
- The first task should be 'context' agent with payload: {"context_id": "<context_id>"} to retrieve cached schema.
- SQL tasks should depend on the context task and use payload: {"query": "<sql question>"}.
- Keep payload concise — agent-specific hints only.
"""

_USER_TEMPLATE = "Query: {query}\nContext ID: {context_id}"

_CORRECTION_TEMPLATE = """Your previous response was not valid JSON. Here is the error:
{error}

Please respond ONLY with valid JSON matching the schema. No markdown. No explanation."""


# ── Internal helpers ──────────────────────────────────────────────────────────

def _extract_json(text: str) -> str:
    """Strip markdown fences and return the first JSON object found."""
    text = re.sub(r"```(?:json)?\s*", "", text).strip()
    text = text.rstrip("`").strip()
    start = text.find("{")
    end   = text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("No JSON object found in LLM response")
    return text[start : end + 1]


def _parse_graph(raw: str) -> TaskGraph:
    """Parse raw LLM output into a validated TaskGraph."""
    cleaned = _extract_json(raw)
    data    = json.loads(cleaned)
    for t in data.get("tasks", []):
        t["agent"] = t["agent"].lower().strip()
    return TaskGraph(**data)


async def _call_stub(query: str, context_id: str) -> str:
    """
    Return a canned TaskGraph JSON without calling any LLM.
    Performs simple keyword matching to pick a realistic agent mix.
    """
    q = query.lower()
    tasks: list[dict] = []

    tasks.append({
        "task_id":     "t01",
        "agent":       "context",
        "description": "Fetch dataset schema and metadata",
        "payload":     {"query": query, "context_id": context_id},
        "depends_on":  [],
    })

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

    intent = f"Analyse query: '{query[:80]}'"
    return json.dumps({"intent": intent, "tasks": tasks})


async def _llm_call(messages: list[dict], settings) -> str:
    provider = settings.llm_provider

    if provider == "stub":
        user_content = next(
            (m["content"] for m in messages if m["role"] == "user"), ""
        )
        m = re.search(r"Query: (.+?)(?:\nContext ID|$)", user_content, re.DOTALL)
        query = m.group(1).strip() if m else user_content
        # Extract context_id from user message
        ctx_match = re.search(r"Context ID: (.+?)$", user_content, re.MULTILINE)
        ctx_id = ctx_match.group(1).strip() if ctx_match else ""
        return await _call_stub(query, ctx_id)

    from langchain_core.messages import AIMessage, HumanMessage, SystemMessage


    logger.info("Orchestrator LLM provider: %s", provider)

    lc_messages = []
    for m in messages:
        if m["role"] == "system":
            lc_messages.append(SystemMessage(content=m["content"]))
        elif m["role"] == "user":
            lc_messages.append(HumanMessage(content=m["content"]))
        elif m["role"] == "assistant":
            lc_messages.append(AIMessage(content=m["content"]))

    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        llm = ChatAnthropic(
            api_key=settings.anthropic_api_key,
            model_name=settings.claude_model,
            temperature=settings.planning_temperature,
        )
    elif provider == "groq":
        from langchain_groq import ChatGroq
        llm = ChatGroq(
            api_key=settings.groq_api_key,
            model_name=settings.groq_model,
            temperature=settings.planning_temperature,
        )
    elif provider == "grok":
        from langchain_openai import ChatOpenAI
        llm = ChatOpenAI(
            api_key=settings.xai_api_key,
            base_url="https://api.x.ai/v1",
            model="grok-2-latest",
            temperature=settings.planning_temperature,
        )
    elif provider == "openai":
        from langchain_openai import ChatOpenAI
        llm = ChatOpenAI(
            api_key=settings.openai_api_key,
            model="gpt-4o",
            temperature=settings.planning_temperature,
        )
    elif provider == "azure_openai":
        from langchain_openai import AzureChatOpenAI
        llm = AzureChatOpenAI(
            api_key=settings.azure_openai_api_key,
            azure_endpoint=settings.azure_openai_endpoint,
            azure_deployment=settings.azure_openai_deployment_name,
            api_version=settings.azure_openai_api_version,
            temperature=settings.planning_temperature,
        )
    else:  # ollama
        from langchain_ollama import ChatOllama
        llm = ChatOllama(
            base_url=settings.ollama_base_url,
            model=settings.ollama_model,
            temperature=settings.planning_temperature,
        )

    response = await llm.ainvoke(lc_messages)
    return response.content


# ── Public planning function ──────────────────────────────────────────────────

async def plan_query(query: str, context_id: str) -> TaskGraph:
    """
    Ask the configured LLM to decompose the query into a TaskGraph.
    Retries once with a correction prompt on JSON parse errors.
    """
    settings = get_settings()

    messages: list[dict[str, str]] = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user",   "content": _USER_TEMPLATE.format(
            query=query, context_id=context_id
        )},
    ]

    raw = ""
    for attempt in range(settings.planning_max_retries + 1):
        try:
            raw = await _llm_call(messages, settings)
            logger.debug("LLM raw output (attempt %d): %.300s", attempt, raw)
            graph = _parse_graph(raw)
            logger.info("Plan produced: %d tasks, intent=%r", len(graph.tasks), graph.intent)
            return graph
        except (json.JSONDecodeError, KeyError, ValueError) as exc:
            if attempt < settings.planning_max_retries:
                logger.warning(
                    "JSON parse error (attempt %d): %s — retrying with correction", attempt, exc
                )
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

    raise RuntimeError("Unexpected exit from planning loop")
