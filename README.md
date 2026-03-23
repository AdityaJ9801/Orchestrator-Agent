# AXIOM AI — Orchestrator Agent

> **Central planner for the GEMRSLIZE multi-agent data analysis platform.**  
> Accepts natural-language queries, decomposes them into a dynamic task graph via an LLM, dispatches tasks to specialist agents in parallel, and returns (or streams) unified analysis results.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Architecture](#2-architecture)
3. [Specialist Agents](#3-specialist-agents)
4. [API Endpoints](#4-api-endpoints)
5. [Data Models](#5-data-models)
6. [LLM Providers](#6-llm-providers)
7. [Configuration & Environment Files](#7-configuration--environment-files)
8. [Project Structure](#8-project-structure)
9. [Quick Start — Local Stub Testing (No Docker, No LLM)](#9-quick-start--local-stub-testing-no-docker-no-llm)
10. [Running with Docker Compose (Free / Ollama Mode)](#10-running-with-docker-compose-free--ollama-mode)
11. [Running in Production (Paid / Claude Mode)](#11-running-in-production-paid--claude-mode)
12. [Testing](#12-testing)
13. [Session Management (Redis)](#13-session-management-redis)
14. [Dependencies](#14-dependencies)
15. [Deployment Notes (EC2)](#15-deployment-notes-ec2)

---

## 1. Project Overview

The **Orchestrator Agent** is the single public-facing entry point of the AXIOM AI platform. It provides a simple HTTP API that:

1. **Receives** a natural-language user query and a `context_id` (a reference to a pre-loaded dataset).
2. **Plans** — calls an LLM (Ollama, Claude, or a local stub) which produces a JSON **TaskGraph**: an ordered, dependency-aware list of specialist agent tasks.
3. **Executes** — runs all tasks in the TaskGraph with maximum parallelism, respecting dependencies. Upstream failure automatically skips dependent tasks.
4. **Returns** — a unified JSON response, or streams real-time progress via **Server-Sent Events (SSE)**.
5. **Persists** — saves the session to **Redis** so follow-up queries can resume context.

---

## 2. Architecture

```
                       ┌─────────────────────────────────────────┐
                       │         Orchestrator Agent  :8000        │
                       │                                           │
  User / Frontend ───► │  POST /analyze   ──► Planner (LLM)       │
                       │  POST /analyze/stream (SSE)              │
                       │  POST /plan      ──► dry-run only         │
                       │  GET  /agents/status                     │
                       │  GET  /health                            │
                       │                                           │
                       │  Executor (asyncio parallel dispatch)     │
                       └────┬──────┬──────┬──────┬──────┬─────────┘
                            │      │      │      │      │
                         :8001  :8002  :8003  :8004  :8005   :8006
                       Context  SQL    Viz    ML    NLP    Report
                        Agent  Agent  Agent  Agent  Agent   Agent
                                          │
                    ┌───────────┐   ┌─────┴──────┐
                    │  Ollama   │   │   Redis     │
                    │  :11434   │   │   :6379     │
                    └───────────┘   └────────────┘
```

**Key design decisions:**

- The Orchestrator is the **only service exposed publicly** (port 8000). All specialist agents are internal.
- Tasks with no mutual dependencies execute **concurrently** using `asyncio.gather`.
- If a dependency fails, all tasks that depend on it are automatically **SKIPPED** (no cascade hang).
- Redis stores session data with a configurable TTL (default 1 hour) for stateful follow-up queries.

---

## 3. Specialist Agents

These are the downstream microservices the Orchestrator dispatches to. Each exposes:
- `GET /health` — liveness probe
- `POST /run` — execute a task with a JSON payload

| Agent Name | Port | Purpose |
|---|---|---|
| `context` | 8001 | Fetches dataset schema, metadata, row counts, column definitions |
| `sql` | 8002 | Executes SQL queries against the data source, returns rows |
| `viz` | 8003 | Generates charts and visualisations (bar, line, trend, etc.) |
| `ml` | 8004 | Trains ML models, generates predictions, reports feature importances |
| `nlp` | 8005 | Performs NLP tasks: sentiment analysis, topic extraction, summarization |
| `report` | 8006 | Aggregates all results and produces a final structured report |

> The Orchestrator's LLM planner always includes a `report` task as the final aggregator in every TaskGraph.

---

## 4. API Endpoints

### `GET /health`
Liveness probe. Returns `200 OK` when the service is running.

```json
{"status": "ok", "service": "orchestrator-agent"}
```

---

### `POST /analyze`
**Full pipeline**: Plan → Execute → Persist → Return.

**Request body:**
```json
{
  "query": "What were total sales by region last quarter? Include a bar chart.",
  "context_id": "ctx-abc-123",
  "session_id": "optional-prior-session-uuid"
}
```

| Field | Type | Required | Description |
|---|---|---|---|
| `query` | string (min 3 chars) | ✅ | Natural-language user question |
| `context_id` | string | ✅ | UUID from Context Agent identifying the dataset |
| `session_id` | string | ❌ | Prior session UUID to resume (for follow-up queries) |

**Response:** `AnalyzeResponse` JSON (see [Data Models](#5-data-models)).

---

### `POST /analyze/stream`
**Same pipeline as `/analyze`**, but returns a **Server-Sent Events** stream so the frontend can show real-time progress.

**SSE event types:**

| Event | When | Payload |
|---|---|---|
| `plan` | Task graph is ready | `{ task_id, intent, graph }` |
| `task_start` | An agent task begins | `{ task_id, node_id, agent, started_at }` |
| `task_complete` | An agent task finishes | `{ task_id, node_id, agent, status, duration_ms, error }` |
| `result` | All tasks done | Full results payload |
| `error` | Unrecoverable failure | `{ task_id, detail }` |

**Client example (Python):**
```python
import httpx

with httpx.stream("POST", "http://localhost:8000/analyze/stream", json={
    "query": "Compare revenue across all regions and show a chart.",
    "context_id": "ctx-demo-001"
}, timeout=120) as r:
    for line in r.iter_lines():
        print(line)
```

---

### `POST /plan`
**Dry-run** — calls the LLM and returns the TaskGraph **without dispatching to any agents**. Useful for inspection and debugging.

**Request body:**
```json
{
  "query": "Predict next quarter revenue by product category.",
  "context_id": "ctx-abc-123"
}
```

**Response:** `TaskGraph` JSON.

---

### `GET /agents/status`
Health-checks all 6 downstream agents **in parallel** and returns their status and latency.

**Response:**
```json
{
  "agents": [
    {"agent": "context", "url": "http://context-agent:8001", "healthy": true, "latency_ms": 12.4},
    {"agent": "sql",     "url": "http://sql-agent:8002",     "healthy": true, "latency_ms": 8.1},
    ...
  ],
  "checked_at": "2026-03-24T00:00:00Z"
}
```

---

### `GET /docs` / `GET /redoc`
Interactive Swagger UI and ReDoc — auto-generated by FastAPI.

---

## 5. Data Models

### `TaskNode`
A single node in the execution graph.
```python
task_id:     str          # short alphanumeric ID (e.g. "t01")
agent:       AgentName    # context | sql | viz | ml | nlp | report
description: str          # what this agent should do
payload:     dict         # agent-specific hints / parameters
depends_on:  list[str]    # task_ids this task waits for
```

### `TaskGraph`
The full LLM-produced execution plan.
```python
intent:     str           # one-sentence summary of the user's goal
tasks:      list[TaskNode]
created_at: datetime
```

### `TaskResult`
Runtime execution state for one node.
```python
task_id:    str
agent:      AgentName
status:     pending | running | completed | failed | skipped
result:     Any           # agent's JSON response
error:      str | None
started_at: datetime | None
ended_at:   datetime | None
duration_ms: float | None  # computed property
```

### `AnalyzeResponse`
The full response from `/analyze`.
```python
task_id:    str           # UUID for this analysis run
session_id: str           # UUID for the session (resumable)
intent:     str
graph:      TaskGraph
results:    list[TaskResult]
partial:    bool          # True if any task failed/was skipped
created_at: datetime
```

---

## 6. LLM Providers

Set via the `LLM_PROVIDER` environment variable:

| Value | Provider | Use case | Cost |
|---|---|---|---|
| `stub` | Local keyword-matching | Local dev/testing — no LLM required | Free |
| `ollama` | Ollama (local Docker) | Full LLM, runs on-prem | Free |
| `claude` | Anthropic Claude | Production, highest quality | Paid |

### Stub Provider
Uses simple keyword heuristics to select agents. No network call. Ideal for running full integration tests without Ollama or an API key.

### Ollama Provider
Calls Ollama's `/api/chat` endpoint. Default model: `qwen3:0.6b`. Configure with:
```
OLLAMA_BASE_URL=http://ollama:11434
OLLAMA_MODEL=qwen3:0.6b   # or llama3.1:8b for better quality
```

### Claude Provider
Uses the `anthropic` Python SDK. Requires an API key. Supports LangSmith tracing.
```
ANTHROPIC_API_KEY=sk-ant-...
CLAUDE_MODEL=claude-sonnet-4-20250514
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=ls__...
```

### Retry Logic
On a JSON parse error from the LLM, the Orchestrator sends one correction prompt and retries. Controlled by `PLANNING_MAX_RETRIES` (default: 1).

---

## 7. Configuration & Environment Files

The Orchestrator uses **`pydantic-settings`** with a layered environment file system. Set `ENV_FILE` to select a profile — this prevents Docker hostnames from overriding local localhost URLs.

| File | Purpose |
|---|---|
| `.env.stub` | Local testing — stub agents on localhost, no LLM |
| `.env.free` | Docker Compose with Ollama (free LLM) |
| `.env.paid` | Production with Claude + LangSmith + EC2 |

### All Configuration Keys

| Variable | Default | Description |
|---|---|---|
| `LLM_PROVIDER` | `ollama` | `stub` / `ollama` / `claude` |
| `OLLAMA_BASE_URL` | `http://ollama:11434` | Ollama service URL |
| `OLLAMA_MODEL` | `qwen3:0.6b` | Ollama model name |
| `ANTHROPIC_API_KEY` | `` | Anthropic API key (Claude mode) |
| `CLAUDE_MODEL` | `claude-sonnet-4-20250514` | Claude model ID |
| `LANGCHAIN_TRACING_V2` | `false` | Enable LangSmith tracing |
| `LANGCHAIN_API_KEY` | `` | LangSmith API key |
| `REDIS_URL` | `redis://redis:6379` | Redis connection string |
| `SESSION_TTL_SECONDS` | `3600` | Session expiry (seconds) |
| `PORT` | `8000` | Service listen port |
| `LOG_LEVEL` | `INFO` | Logging verbosity |
| `USE_DOCKER` | `true` | Use Docker service hostnames |
| `USE_EC2` | `false` | Use EC2 private IP overrides |
| `AGENT_TIMEOUT_SECONDS` | `30.0` | Max wait per agent call |
| `PLANNING_MAX_RETRIES` | `1` | LLM JSON parse retry count |
| `PLANNING_TEMPERATURE` | `0.2` | LLM sampling temperature |
| `CONTEXT_AGENT_URL` | `http://context-agent:8001` | Context agent URL |
| `SQL_AGENT_URL` | `http://sql-agent:8002` | SQL agent URL |
| `VIZ_AGENT_URL` | `http://viz-agent:8003` | Viz agent URL |
| `ML_AGENT_URL` | `http://ml-agent:8004` | ML agent URL |
| `NLP_AGENT_URL` | `http://nlp-agent:8005` | NLP agent URL |
| `REPORT_AGENT_URL` | `http://report-agent:8006` | Report agent URL |
| `EC2_CONTEXT_AGENT_IP` | `` | EC2 private IP for context agent |
| `EC2_SQL_AGENT_IP` | `` | EC2 private IP for SQL agent |
| `EC2_VIZ_AGENT_IP` | `` | EC2 private IP for viz agent |
| `EC2_ML_AGENT_IP` | `` | EC2 private IP for ML agent |
| `EC2_NLP_AGENT_IP` | `` | EC2 private IP for NLP agent |
| `EC2_REPORT_AGENT_IP` | `` | EC2 private IP for report agent |

---

## 8. Project Structure

```
Orchestrator/
│
├── app/                          # Core application package
│   ├── main.py                   # FastAPI app factory + lifespan
│   ├── config.py                 # pydantic-settings configuration
│   ├── models.py                 # Pydantic request/response models
│   ├── planner.py                # LLM integration — produces TaskGraph
│   ├── executor.py               # Async parallel execution engine
│   ├── session_store.py          # Redis-backed session persistence
│   └── routes/
│       ├── analyze.py            # POST /analyze
│       ├── stream.py             # POST /analyze/stream (SSE)
│       ├── plan.py               # POST /plan (dry-run)
│       └── agents.py             # GET /agents/status
│
├── stubs/                        # Local testing tools
│   ├── mock_agent.py             # Fake specialist agent (GET /health, POST /run)
│   ├── run_stubs.py              # Launches all 6 mock agents in parallel
│   ├── demo_client.py            # Demo HTTP client for quick API exploration
│   └── scenario_test.py          # 10-scenario integration test suite
│
├── tests/                        # Pytest unit tests
│   ├── test_planner.py           # Unit tests for planner.py
│   └── test_executor.py          # Unit tests for executor.py
│
├── .env.stub                     # Env config for local stub testing
├── .env.free                     # Env config for Docker + Ollama
├── .env.paid                     # Env config for production (Claude + EC2)
├── Dockerfile                    # Container image definition
├── docker-compose.yml            # Full stack: Orchestrator + Ollama + Redis
├── requirements.txt              # Python dependencies
└── pyproject.toml                # pytest configuration
```

---

## 9. Quick Start — Local Stub Testing (No Docker, No LLM)

This is the fastest way to develop and test. No Docker, no Ollama, no API key needed.

### Prerequisites
- Python 3.11+
- Redis running locally on port 6379 (or skip — Redis failures are non-fatal)

### Step 1: Install dependencies
```bash
pip install -r requirements.txt
pip install -e .
```

### Step 2: Start the 6 mock agent stubs
Open **Terminal 1:**
```bash
python stubs/run_stubs.py
```
This launches all 6 fake agents on ports 8001–8006. Press `Ctrl+C` to stop.

### Step 3: Start the Orchestrator (stub mode)
Open **Terminal 2** (PowerShell):
```powershell
$env:ENV_FILE=".env.stub"
uvicorn app.main:app --port 8000 --reload
```

Or on Linux/macOS:
```bash
ENV_FILE=.env.stub uvicorn app.main:app --port 8000 --reload
```

### Step 4: Run the scenario tests
Open **Terminal 3:**
```bash
python stubs/scenario_test.py
```

Run a specific scenario only:
```bash
python stubs/scenario_test.py --scenario 2
```

### Step 5: Explore the API
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc
- Health: http://localhost:8000/health
- Agent status: http://localhost:8000/agents/status

### Quick manual test
```bash
curl -X POST http://localhost:8000/plan \
  -H "Content-Type: application/json" \
  -d '{"query": "Show total revenue by region for last quarter.", "context_id": "ctx-demo"}'
```

---

## 10. Running with Docker Compose (Free / Ollama Mode)

Runs the full stack locally: Orchestrator + Ollama (local LLM) + Redis + all specialist agents.

```bash
# Build and start
docker compose up --build

# Pull the Ollama model (first time only)
docker exec -it <ollama-container-id> ollama pull qwen3:0.6b

# The Orchestrator is now live at:
# http://localhost:8000
```

> **Note:** Ollama requires ~5 GB of disk and significant RAM. Use `qwen3:0.6b` for minimal resource usage, or `llama3.1:8b` for better planning quality.

### docker-compose.yml services

| Service | Port | Description |
|---|---|---|
| `ollama` | 11434 | Local LLM inference engine |
| `redis` | 6379 | Session store |
| `orchestrator` | 8000 | This service (only public port) |
| `context-agent` | 8001 (internal) | Context specialist |
| `sql-agent` | 8002 (internal) | SQL specialist |
| `viz-agent` | 8003 (internal) | Viz specialist |
| `ml-agent` | 8004 (internal) | ML specialist |
| `nlp-agent` | 8005 (internal) | NLP specialist |
| `report-agent` | 8006 (internal) | Report specialist |

---

## 11. Running in Production (Paid / Claude Mode)

Fill in your secrets in `.env.paid`, then deploy:

```bash
# .env.paid (edit these values)
LLM_PROVIDER=claude
ANTHROPIC_API_KEY=sk-ant-YOUR_KEY_HERE
CLAUDE_MODEL=claude-sonnet-4-20250514
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=ls__YOUR_KEY_HERE
USE_EC2=true
EC2_CONTEXT_AGENT_IP=10.0.1.10
# ... (fill all EC2 IPs)
```

For EC2 deployments, set `USE_EC2=true` and provide each agent's private IP via `EC2_*_AGENT_IP` variables. The `agent_registry` property in `Settings` automatically resolves to `http://<ip>:<port>` for EC2 mode.

---

## 12. Testing

### Unit Tests (pytest)
```bash
pytest
```

Tests cover:
- **`tests/test_planner.py`** — LLM provider switching, JSON extraction, retry logic, stub graph generation
- **`tests/test_executor.py`** — Parallel task dispatch, dependency resolution, failure/skip propagation, timeout handling

### Integration / Scenario Tests
Run against a live Orchestrator + stubs:

```bash
python stubs/scenario_test.py
```

The 10 scenarios cover:

| # | Scenario | Endpoint |
|---|---|---|
| 1 | Simple SQL query | `POST /plan` |
| 2 | Multi-agent full execution | `POST /analyze` |
| 3 | ML prediction query | `POST /plan` |
| 4 | NLP / sentiment query | `POST /plan` |
| 5 | Session resume (follow-up) | `POST /analyze` |
| 6 | Validation error (too-short query) | `POST /analyze` |
| 7 | Unknown context_id (graceful) | `POST /analyze` |
| 8 | Plan-only dry run | `POST /plan` |
| 9 | SSE streaming event sequence | `POST /analyze/stream` |
| 10 | All-agents health check | `GET /agents/status` |

Run a single scenario:
```bash
python stubs/scenario_test.py --scenario 9
```

---

## 13. Session Management (Redis)

Each successful `/analyze` or `/analyze/stream` call persists the full result payload to Redis under the key:
```
orchestrator:session:<session_id>
```

**TTL:** Configurable via `SESSION_TTL_SECONDS` (default: `3600` = 1 hour).

To **resume a prior session**, pass the `session_id` from a previous response:
```json
{
  "query": "Now drill down into the South region.",
  "context_id": "ctx-abc-123",
  "session_id": "xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx"
}
```

Redis failures are **non-fatal** — the Orchestrator logs a warning and returns the result anyway.

---

## 14. Dependencies

| Package | Purpose |
|---|---|
| `fastapi` | Web framework |
| `uvicorn[standard]` | ASGI server |
| `pydantic` | Data validation |
| `pydantic-settings` | Environment config |
| `httpx` | Async HTTP client (agent calls) |
| `redis[asyncio]` | Async Redis client |
| `anthropic` | Claude SDK (paid mode) |
| `ollama` | Ollama SDK (free mode) |
| `tenacity` | Retry logic |
| `langsmith` | LLM tracing (paid mode) |
| `pytest` | Unit testing |
| `pytest-asyncio` | Async test support |

Install all:
```bash
pip install -r requirements.txt
```

---

## 15. Deployment Notes (EC2)

- Deploy each specialist agent on its own EC2 instance (or in a private subnet).
- Set `USE_EC2=true` and provide `EC2_*_AGENT_IP` private IP addresses.
- Only the **Orchestrator** (port 8000) needs to be in a public subnet / behind a load balancer.
- All inter-agent communication stays on the private network.
- The Dockerfile includes a `HEALTHCHECK` on `GET /health` — use this with EC2 target group health checks.

```dockerfile
HEALTHCHECK --interval=15s --timeout=5s --start-period=10s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1
```

---

## License

Internal project — AXIOM AI / GEMRSLIZE platform.  
© 2026 All rights reserved.
