"""
scenario_test.py
================
Scenario-based integration tests for the Orchestrator Agent API.
Tests specific real-world situations to verify correct processing.

Scenarios covered:
  1. Simple single-intent query   (SQL only)
  2. Multi-agent complex query     (context → sql → viz → report)
  3. ML prediction query           (context → sql → ml → report)
  4. NLP / sentiment query         (context → nlp → report)
  5. Session resume (follow-up)    (reuse session_id from prior run)
  6. Empty / very short query      (expect 422 validation error)
  7. Unknown context_id            (should still plan + execute gracefully)
  8. Plan-only dry run             (POST /plan, no agents called)
  9. SSE streaming scenario        (verify event sequence from /analyze/stream)
 10. Agents health check           (GET /agents/status, verify all 6 expected)

Prerequisites:
  • stubs running:       python stubs/run_stubs.py
  • orchestrator up:     $env:ENV_FILE=".env.stub"; uvicorn app.main:app --port 8000 --reload
  • redis:               optional (failures tolerated)

Usage:
    python stubs/scenario_test.py [--base http://localhost:8000] [--scenario N]

    --scenario N   run only scenario N (1-10); omit to run all
"""
from __future__ import annotations

import argparse
import json
import sys
import textwrap
import time
from typing import Optional

import httpx

# Force UTF-8 output on Windows to avoid cp1252 encoding errors
import sys as _sys
if hasattr(_sys.stdout, "reconfigure"):
    try:
        _sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore
    except Exception:
        pass

# ── ANSI colours ──────────────────────────────────────────────────────────────
GREEN  = "\033[92m"
YELLOW = "\033[93m"
RED    = "\033[91m"
CYAN   = "\033[96m"
MAGENTA = "\033[95m"
BOLD   = "\033[1m"
DIM    = "\033[2m"
RESET  = "\033[0m"

PASS = f"{GREEN}[PASS]{RESET}"
FAIL = f"{RED}[FAIL]{RESET}"
WARN = f"{YELLOW}[WARN]{RESET}"
INFO = f"{CYAN}[INFO]{RESET}"

_total  = 0
_passed = 0
_failed = 0


def _section(n: int, title: str) -> None:
    print(f"\n{BOLD}{CYAN}{'-'*62}")
    print(f"  Scenario {n:02d} -- {title}")
    print(f"{'-'*62}{RESET}")


def _check(label: str, condition: bool, detail: str = "") -> None:
    global _total, _passed, _failed
    _total += 1
    if condition:
        _passed += 1
        print(f"  {PASS}  {label}" + (f"  {DIM}({detail}){RESET}" if detail else ""))
    else:
        _failed += 1
        print(f"  {FAIL}  {label}" + (f"  {DIM}({detail}){RESET}" if detail else ""))


def _info(msg: str) -> None:
    print(f"  {INFO}  {msg}")


def _json_snippet(obj: dict, max_keys: int = 6) -> str:
    keys = list(obj.keys())[:max_keys]
    brief = {k: obj[k] for k in keys}
    return json.dumps(brief, default=str)


# ── Helper: safe POST ─────────────────────────────────────────────────────────

def _post(client: httpx.Client, url: str, payload: dict,
          timeout: float = 90) -> tuple[int, dict]:
    try:
        r = client.post(url, json=payload, timeout=timeout)
        try:
            body = r.json()
        except Exception:
            body = {"_raw": r.text[:300]}
        return r.status_code, body
    except httpx.ConnectError as exc:
        return 0, {"_error": str(exc)}
    except Exception as exc:
        return -1, {"_error": str(exc)}


def _get(client: httpx.Client, url: str, timeout: float = 15) -> tuple[int, dict]:
    try:
        r = client.get(url, timeout=timeout)
        return r.status_code, r.json()
    except Exception as exc:
        return -1, {"_error": str(exc)}


# ─────────────────────────────────────────────────────────────────────────────
#  SCENARIO 1: Simple single-agent query (SQL only)
# ─────────────────────────────────────────────────────────────────────────────

def scenario_01_simple_sql(base: str, client: httpx.Client) -> Optional[str]:
    _section(1, "Simple SQL-only query → /plan should produce sql + report tasks")
    payload = {
        "query":      "Show me total revenue for last month grouped by region.",
        "context_id": "ctx-simple-sql-001",
    }
    status, body = _post(client, f"{base}/plan", payload)
    _check("HTTP 200 from /plan", status == 200, f"got {status}")
    if status != 200:
        _info(f"Response: {body}")
        return None

    tasks = body.get("tasks", [])
    agents = [t["agent"] for t in tasks]
    _check("At least 2 tasks in graph", len(tasks) >= 2, f"{len(tasks)} tasks")
    _check("'sql' agent present",    "sql"    in agents, f"agents={agents}")
    _check("'report' agent present", "report" in agents, f"agents={agents}")

    # report must depend on sql
    report_task = next((t for t in tasks if t["agent"] == "report"), None)
    if report_task:
        _check(
            "'report' depends on 'sql'",
            any(dep in [t["task_id"] for t in tasks if t["agent"] == "sql"]
                for dep in report_task.get("depends_on", [])),
            f"depends_on={report_task.get('depends_on')}",
        )
    _info(f"Intent: {body.get('intent', '?')}")
    return None


# ─────────────────────────────────────────────────────────────────────────────
#  SCENARIO 2: Complex multi-agent query (context → sql → viz → report)
# ─────────────────────────────────────────────────────────────────────────────

def scenario_02_multi_agent(base: str, client: httpx.Client) -> Optional[str]:
    _section(2, "Complex multi-agent analysis → /analyze full execution")
    payload = {
        "query": (
            "What were total sales by region last quarter? "
            "Include a bar chart and a comprehensive summary report."
        ),
        "context_id": "ctx-multi-agent-002",
    }
    t0 = time.monotonic()
    status, body = _post(client, f"{base}/analyze", payload, timeout=120)
    elapsed_ms = (time.monotonic() - t0) * 1000

    _check("HTTP 200 from /analyze", status == 200, f"got {status}")
    if status != 200:
        _info(f"Response: {body}")
        return None

    results  = body.get("results", [])
    statuses = {r["agent"]: r["status"] for r in results}
    session_id = body.get("session_id")

    _check("session_id returned",      bool(session_id))
    _check("intent not empty",         bool(body.get("intent")))
    _check("at least 3 tasks ran",     len(results) >= 3, f"{len(results)} tasks")
    _check("no task FAILED or SKIPPED",
           all(s == "completed" for s in statuses.values()),
           f"statuses={statuses}")
    _check("'report' task completed",  statuses.get("report") == "completed")
    _check("finished in < 30 s",       elapsed_ms < 30_000, f"{elapsed_ms:.0f} ms")

    _info(f"Intent: {body.get('intent')}")
    _info(f"Elapsed: {elapsed_ms:.0f} ms  |  Tasks: {len(results)}")
    for r in results:
        dur = r.get("duration_ms") or 0
        print(f"       {r['status']:10}  [{r['task_id']}] {r['agent']:8}  {dur:.0f} ms")

    return session_id


# ─────────────────────────────────────────────────────────────────────────────
#  SCENARIO 3: ML prediction / forecasting query
# ─────────────────────────────────────────────────────────────────────────────

def scenario_03_ml_query(base: str, client: httpx.Client) -> None:
    _section(3, "ML prediction query → expects 'ml' agent in task graph")
    payload = {
        "query": (
            "Predict next quarter's revenue for each product category "
            "using historical sales data. Show feature importances."
        ),
        "context_id": "ctx-ml-003",
    }
    status, body = _post(client, f"{base}/plan", payload)
    _check("HTTP 200 from /plan", status == 200, f"got {status}")
    if status != 200:
        return

    tasks  = body.get("tasks", [])
    agents = [t["agent"] for t in tasks]
    _check("'ml' agent in task graph",   "ml" in agents,     f"agents={agents}")
    _check("'report' agent in graph",    "report" in agents, f"agents={agents}")
    _check("'sql' or 'context' present", ("sql" in agents or "context" in agents),
           f"agents={agents}")

    _info(f"Intent: {body.get('intent')}")
    _info(f"Task chain: {' → '.join(agents)}")


# ─────────────────────────────────────────────────────────────────────────────
#  SCENARIO 4: NLP / sentiment analysis query
# ─────────────────────────────────────────────────────────────────────────────

def scenario_04_nlp_query(base: str, client: httpx.Client) -> None:
    _section(4, "NLP / sentiment query → expects 'nlp' agent in task graph")
    payload = {
        "query": (
            "Analyse customer feedback from last month. "
            "Summarise the main topics, entities, and overall sentiment."
        ),
        "context_id": "ctx-nlp-004",
    }
    status, body = _post(client, f"{base}/plan", payload)
    _check("HTTP 200 from /plan", status == 200, f"got {status}")
    if status != 200:
        return

    tasks  = body.get("tasks", [])
    agents = [t["agent"] for t in tasks]
    _check("'nlp' agent in task graph",  "nlp"    in agents, f"agents={agents}")
    _check("'report' agent in graph",    "report" in agents, f"agents={agents}")

    _info(f"Intent: {body.get('intent')}")


# ─────────────────────────────────────────────────────────────────────────────
#  SCENARIO 5: Session resume (follow-up query reuses session_id)
# ─────────────────────────────────────────────────────────────────────────────

def scenario_05_session_resume(base: str, client: httpx.Client,
                                prior_session_id: Optional[str]) -> None:
    _section(5, "Session resume — follow-up query passes prior session_id")

    if not prior_session_id:
        # Run a quick analyse to get a valid session first
        _info("No prior session available — running a quick /analyze to obtain one")
        _, body = _post(client, f"{base}/analyze", {
            "query":      "List total revenue by region.",
            "context_id": "ctx-resume-seed-005",
        }, timeout=90)
        prior_session_id = body.get("session_id")

    if not prior_session_id:
        _check("session_id obtained for resume", False, "could not get session_id")
        return

    _info(f"Resuming session: {prior_session_id}")
    payload = {
        "query":      "Now drill down into the South region. What drove its top performance?",
        "context_id": "ctx-resume-005",
        "session_id": prior_session_id,
    }
    status, body = _post(client, f"{base}/analyze", payload, timeout=120)
    _check("HTTP 200 on follow-up /analyze", status == 200, f"got {status}")
    if status != 200:
        _info(f"Body: {body}")
        return

    _check("session_id returned",        bool(body.get("session_id")))
    _check("intent not empty",           bool(body.get("intent")))
    _check("results list returned",      isinstance(body.get("results"), list))
    _info(f"Follow-up intent: {body.get('intent')}")


# ─────────────────────────────────────────────────────────────────────────────
#  SCENARIO 6: Invalid / too-short query → expect 422
# ─────────────────────────────────────────────────────────────────────────────

def scenario_06_validation_error(base: str, client: httpx.Client) -> None:
    _section(6, "Validation: query too short → expect HTTP 422")
    payload = {"query": "Hi", "context_id": "ctx-bad-006"}   # min_length=3 → "Hi" is 2 chars
    status, body = _post(client, f"{base}/analyze", payload)
    _check("HTTP 422 returned for too-short query", status == 422,
           f"got {status}  body={str(body)[:120]}")

    _section(6, "Validation: missing query field → expect HTTP 422")
    status2, body2 = _post(client, f"{base}/analyze", {"context_id": "ctx-bad-006b"})
    _check("HTTP 422 returned for missing query", status2 == 422,
           f"got {status2}  body={str(body2)[:120]}")


# ─────────────────────────────────────────────────────────────────────────────
#  SCENARIO 7: Unknown / synthetic context_id (system should still work)
# ─────────────────────────────────────────────────────────────────────────────

def scenario_07_unknown_context(base: str, client: httpx.Client) -> None:
    _section(7, "Unknown context_id — Orchestrator should still plan & execute")
    payload = {
        "query":      "Summarise the dataset schema and key statistics.",
        "context_id": "ctx-totally-unknown-xyz-999",
    }
    status, body = _post(client, f"{base}/analyze", payload, timeout=90)
    _check("HTTP 200 returned (no crash)", status == 200, f"got {status}")
    if status == 200:
        _check("results list returned", isinstance(body.get("results"), list))
        _info(f"Intent: {body.get('intent')}")


# ─────────────────────────────────────────────────────────────────────────────
#  SCENARIO 8: Plan-only dry run → POST /plan, verify structure
# ─────────────────────────────────────────────────────────────────────────────

def scenario_08_plan_dry_run(base: str, client: httpx.Client) -> None:
    _section(8, "Plan dry-run → POST /plan validates TaskGraph schema")
    payload = {
        "query":      "Which product categories have declining sales over the past 6 months? "
                      "Show trend lines and forecast next month.",
        "context_id": "ctx-plan-dry-008",
    }
    t0 = time.monotonic()
    status, body = _post(client, f"{base}/plan", payload, timeout=60)
    elapsed_ms = (time.monotonic() - t0) * 1000

    _check("HTTP 200 from /plan", status == 200, f"got {status}")
    if status != 200:
        _info(str(body)[:200])
        return

    tasks      = body.get("tasks", [])
    task_ids   = {t["task_id"] for t in tasks}
    agents     = [t["agent"] for t in tasks]
    all_deps   = [dep for t in tasks for dep in t.get("depends_on", [])]

    _check("Non-empty task list",          len(tasks) > 0,       f"{len(tasks)} tasks")
    _check("All dep IDs exist in graph",   all(d in task_ids for d in all_deps),
           f"missing={[d for d in all_deps if d not in task_ids]}")
    _check("'report' is last agent",       tasks[-1]["agent"] == "report" if tasks else False)
    _check("intent field present",         bool(body.get("intent")))
    _check("Plan returned in < 60 s",      elapsed_ms < 60_000, f"{elapsed_ms:.0f} ms")

    _info(f"Intent: {body.get('intent')}")
    _info(f"Agent chain: {' → '.join(agents)}")


# ─────────────────────────────────────────────────────────────────────────────
#  SCENARIO 9: SSE streaming → /analyze/stream event sequence
# ─────────────────────────────────────────────────────────────────────────────

def scenario_09_sse_stream(base: str) -> None:
    _section(9, "SSE Streaming → POST /analyze/stream event sequence")
    payload = {
        "query":      "Compare revenue across all regions and visualise with a chart.",
        "context_id": "ctx-stream-009",
    }
    events_seen: list[str] = []
    data_lines  = 0
    error_flag  = False

    print(f"  {INFO}  Streaming events:")
    try:
        with httpx.stream(
            "POST",
            f"{base}/analyze/stream",
            json=payload,
            timeout=120,
        ) as response:
            if response.status_code != 200:
                _check("HTTP 200 from /analyze/stream", False,
                       f"got {response.status_code}")
                return
            for line in response.iter_lines():
                line = line.strip()
                if not line:
                    continue
                if line.startswith("event:"):
                    ev = line.split(":", 1)[1].strip()
                    events_seen.append(ev)
                    colour = {
                        "plan": CYAN, "task_start": "", "task_complete": GREEN,
                        "result": BOLD + GREEN, "error": RED,
                    }.get(ev, "")
                    print(f"       {colour}>> {ev}{RESET}")
                    if ev == "error":
                        error_flag = True
                elif line.startswith("data:"):
                    data_lines += 1
    except KeyboardInterrupt:
        print(f"  {WARN}  Stream interrupted by user")
    except Exception as exc:
        _check("Stream completed without exception", False, str(exc))
        return

    _check("HTTP 200 from /analyze/stream",     True)   # we got here
    _check("'plan' event received",             "plan"          in events_seen, f"{events_seen}")
    _check("'task_start' events received",      "task_start"    in events_seen, f"{events_seen}")
    _check("'task_complete' events received",   "task_complete" in events_seen, f"{events_seen}")
    _check("'result' event received",           "result"        in events_seen, f"{events_seen}")
    _check("No 'error' events",                 not error_flag,                 f"{events_seen}")
    _check("Data lines sent",                   data_lines > 0,                 f"{data_lines}")
    _info(f"Unique events: {list(dict.fromkeys(events_seen))}")


# ─────────────────────────────────────────────────────────────────────────────
#  SCENARIO 10: Agents health check
# ─────────────────────────────────────────────────────────────────────────────

def scenario_10_agents_status(base: str, client: httpx.Client) -> None:
    _section(10, "GET /agents/status → all 6 specialist agents reported")
    status, body = _get(client, f"{base}/agents/status", timeout=20)
    _check("HTTP 200 from /agents/status", status == 200, f"got {status}")
    if status != 200:
        return

    agents_list  = body.get("agents", [])
    agent_names  = {a["agent"] for a in agents_list}
    expected     = {"context", "sql", "viz", "ml", "nlp", "report"}

    _check("All 6 agents reported",      expected == agent_names,
           f"reported={agent_names}")
    _check("'checked_at' field present", bool(body.get("checked_at")))

    healthy_count = sum(1 for a in agents_list if a.get("healthy"))
    _check(f"All 6 agents healthy ({healthy_count}/6)",
           healthy_count == 6, f"{healthy_count}/6 healthy")

    print(f"\n  {'Agent':<12} {'Healthy':^8} {'Latency ms':>12}  Error")
    print(f"  {'-'*50}")
    for a in sorted(agents_list, key=lambda x: x["agent"]):
        col  = GREEN if a["healthy"] else RED
        lat  = f"{a.get('latency_ms', '?'):.1f}" if isinstance(a.get("latency_ms"), float) else "?"
        err  = a.get("error") or ""
        mark = "[ok]" if a["healthy"] else "[--]"
        print(f"  {col}{a['agent']:<12}{RESET} {mark:^8} {lat:>12}  {err}")


# ─────────────────────────────────────────────────────────────────────────────
#  Summary
# ─────────────────────────────────────────────────────────────────────────────

def _print_summary() -> None:
    print(f"\n{BOLD}{'='*62}")
    print(f"  SCENARIO TEST SUMMARY")
    print(f"{'='*62}{RESET}  ")
    colour = GREEN if _failed == 0 else RED
    print(f"  {colour}Passed : {_passed}/{_total}{RESET}")
    if _failed:
        print(f"  {RED}Failed : {_failed}/{_total}{RESET}")
    print(f"  {DIM}Checks : {_total} total{RESET}")
    print(f"{'='*62}\n")


# ─────────────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Orchestrator Scenario Test Runner")
    parser.add_argument("--base", default="http://localhost:8000",
                        help="Orchestrator base URL (default: http://localhost:8000)")
    parser.add_argument("--scenario", type=int, default=0,
                        help="Run only this scenario number (1-10); 0 = all")
    args   = parser.parse_args()
    base   = args.base.rstrip("/")
    only   = args.scenario

    print(f"\n{BOLD}{'='*62}")
    print("  AXIOM AI -- Orchestrator Scenario Test Suite")
    print(f"  Target : {base}")
    print(f"  Time   : {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*62}{RESET}")

    # Quick connectivity check
    try:
        r = httpx.get(f"{base}/health", timeout=5)
        r.raise_for_status()
        print(f"\n  {PASS}  Orchestrator reachable — {r.json()}")
    except Exception as exc:
        print(f"\n  {FAIL}  Cannot reach Orchestrator at {base}: {exc}")
        print("         Ensure uvicorn is running, then retry.")
        sys.exit(1)

    session_from_s2: Optional[str] = None

    with httpx.Client() as client:
        if only in (0, 1):
            scenario_01_simple_sql(base, client)
        if only in (0, 2):
            session_from_s2 = scenario_02_multi_agent(base, client)
        if only in (0, 3):
            scenario_03_ml_query(base, client)
        if only in (0, 4):
            scenario_04_nlp_query(base, client)
        if only in (0, 5):
            scenario_05_session_resume(base, client, session_from_s2)
        if only in (0, 6):
            scenario_06_validation_error(base, client)
        if only in (0, 7):
            scenario_07_unknown_context(base, client)
        if only in (0, 8):
            scenario_08_plan_dry_run(base, client)
        if only in (0, 10):
            scenario_10_agents_status(base, client)

    # SSE scenario uses its own streaming client
    if only in (0, 9):
        scenario_09_sse_stream(base)

    _print_summary()
    sys.exit(1 if _failed else 0)


if __name__ == "__main__":
    main()
