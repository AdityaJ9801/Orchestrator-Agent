"""
demo_client.py
==============
End-to-end smoke test of the Orchestrator Agent.
Requires:
  • stubs running:      python stubs/run_stubs.py
  • orchestrator up:    uvicorn app.main:app --env-file .env.stub --port 8000
  • redis reachable:    (optional — failures are non-fatal)

Usage:
    python stubs/demo_client.py [--base http://localhost:8000]
"""
from __future__ import annotations

import argparse
import json
import sys
import textwrap
import time

import httpx

# ── Formatting helpers ────────────────────────────────────────────────────────

GREEN  = "\033[92m"
YELLOW = "\033[93m"
RED    = "\033[91m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
RESET  = "\033[0m"

DEMO_QUERY      = "What were total sales by region last quarter? Include a bar chart and a summary report."
DEMO_CONTEXT_ID = "ctx-demo-2024-q4"


def _ok(msg: str)   -> str: return f"{GREEN}[OK]  {msg}{RESET}"
def _warn(msg: str) -> str: return f"{YELLOW}[WARN] {msg}{RESET}"
def _err(msg: str)  -> str: return f"{RED}[ERR] {msg}{RESET}"
def _head(msg: str) -> str: return f"\n{BOLD}{CYAN}{'='*60}\n  {msg}\n{'='*60}{RESET}"
def _json(obj: dict) -> str:
    return textwrap.indent(json.dumps(obj, indent=2, default=str), "    ")


# ── Individual demo steps ─────────────────────────────────────────────────────

def demo_health(base: str, client: httpx.Client) -> bool:
    print(_head("1 / 5  — GET /health"))
    try:
        r = client.get(f"{base}/health", timeout=5)
        r.raise_for_status()
        print(_ok(f"Orchestrator is UP — {r.json()}"))
        return True
    except Exception as exc:
        print(_err(f"Orchestrator not reachable: {exc}"))
        print("       Make sure you ran:  uvicorn app.main:app --env-file .env.stub --port 8000")
        return False


def demo_agents_status(base: str, client: httpx.Client) -> None:
    print(_head("2 / 5  — GET /agents/status"))
    try:
        r = client.get(f"{base}/agents/status", timeout=15)
        r.raise_for_status()
        data = r.json()
        for a in data["agents"]:
            name      = a["agent"]
            healthy   = a["healthy"]
            latency   = a.get("latency_ms", "?")
            err       = a.get("error", "")
            icon      = "[ok] " if healthy else "[err]"
            colour    = GREEN if healthy else RED
            print(f"   {colour}{icon} {name:<10} {latency:>7.1f} ms{RESET}"
                  + (f"  -- {err}" if err else ""))
    except Exception as exc:
        print(_warn(f"Could not reach /agents/status: {exc}"))


def demo_plan(base: str, client: httpx.Client) -> None:
    print(_head("3 / 5  — POST /plan  (dry run, no agents called)"))
    payload = {"query": DEMO_QUERY, "context_id": DEMO_CONTEXT_ID}
    try:
        r = client.post(f"{base}/plan", json=payload, timeout=60)
        r.raise_for_status()
        graph = r.json()
        print(_ok(f"TaskGraph produced — intent: {graph.get('intent','?')}"))
        print(f"   Tasks ({len(graph.get('tasks', []))}):")
        for t in graph.get("tasks", []):
            deps = t.get("depends_on", [])
            print(f"     [{t['task_id']}] {t['agent']:10} <- deps={deps}")
    except Exception as exc:
        print(_warn(f"/plan call failed: {exc}"))


def demo_analyze(base: str, client: httpx.Client) -> str | None:
    print(_head("4 / 5  — POST /analyze  (full execution)"))
    payload = {"query": DEMO_QUERY, "context_id": DEMO_CONTEXT_ID}
    try:
        t0 = time.monotonic()
        r  = client.post(f"{base}/analyze", json=payload, timeout=120)
        elapsed = (time.monotonic() - t0) * 1000
        r.raise_for_status()
        data = r.json()

        task_id    = data.get("task_id")
        session_id = data.get("session_id")
        intent     = data.get("intent", "?")
        partial    = data.get("partial", False)
        results    = data.get("results", [])

        print(_ok(f"Analysis complete in {elapsed:.0f} ms"))
        print(f"   task_id    : {task_id}")
        print(f"   session_id : {session_id}")
        print(f"   intent     : {intent}")
        print(f"   partial    : {partial}")
        print(f"\n   Task results:")
        for tr in results:
            status = tr["status"]
            colour = GREEN if status == "completed" else (YELLOW if status == "skipped" else RED)
            dur    = tr.get("duration_ms") or 0
            print(f"     {colour}{status:<10}{RESET}  [{tr['task_id']}] {tr['agent']:<10}  {dur:.0f} ms")

        return session_id
    except Exception as exc:
        print(_warn(f"/analyze call failed: {exc}"))
        return None


def demo_stream(base: str) -> None:
    print(_head("5 / 5  — POST /analyze/stream  (SSE streaming)"))
    payload = {"query": DEMO_QUERY, "context_id": DEMO_CONTEXT_ID}
    print(f"   Streaming events (press Ctrl+C to stop early):\n")
    try:
        with httpx.stream(
            "POST",
            f"{base}/analyze/stream",
            json=payload,
            timeout=120,
        ) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                line = line.strip()
                if not line:
                    continue
                if line.startswith("event:"):
                    event_name = line.split(":", 1)[1].strip()
                    colour = {
                        "plan":          CYAN,
                        "task_start":    "",
                        "task_complete": GREEN,
                        "result":        BOLD + GREEN,
                        "error":         RED,
                    }.get(event_name, "")
                    print(f"   {colour}>> {event_name}{RESET}")
                elif line.startswith("data:"):
                    raw = line[5:].strip()
                    try:
                        obj = json.loads(raw)
                        # Only print small / relevant keys
                        brief = {
                            k: v for k, v in obj.items()
                            if k in ("intent", "status", "agent", "node_id",
                                     "duration_ms", "partial", "task_id", "error")
                        }
                        if brief:
                            print(f"      {json.dumps(brief, default=str)}")
                    except Exception:
                        print(f"      {raw[:120]}")
        print(f"\n   {_ok('Stream complete')}")
    except KeyboardInterrupt:
        print(f"\n   {_warn('Stream interrupted by user')}")
    except Exception as exc:
        print(_warn(f"Stream error: {exc}"))


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Orchestrator demo client")
    parser.add_argument("--base", default="http://localhost:8000",
                        help="Orchestrator base URL (default: http://localhost:8000)")
    args = parser.parse_args()
    base = args.base.rstrip("/")

    print(f"\n{BOLD}{'='*60}")
    print("  GEMRSLIZE — Orchestrator Agent Demo Client")
    print(f"  Target: {base}")
    print(f"{'='*60}{RESET}")

    with httpx.Client() as client:
        if not demo_health(base, client):
            sys.exit(1)
        demo_agents_status(base, client)
        demo_plan(base, client)
        demo_analyze(base, client)

    # SSE uses its own streaming client context
    demo_stream(base)

    print(f"\n{BOLD}{GREEN}{'='*60}")
    print("  All demo steps finished!")
    print(f"{'='*60}{RESET}\n")


if __name__ == "__main__":
    main()
