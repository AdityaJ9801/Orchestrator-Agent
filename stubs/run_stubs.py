"""
run_stubs.py
============
Launches all 6 mock agent stubs in parallel subprocesses.
Run this BEFORE starting the Orchestrator.

  python stubs/run_stubs.py

Stop with  Ctrl+C  — all child processes are cleaned up automatically.
"""
from __future__ import annotations

import signal
import subprocess
import sys
import time
from pathlib import Path

AGENTS = {
    "context": 8001,
    "sql":     8002,
    "viz":     8003,
    "ml":      8004,
    "nlp":     8005,
    "report":  8006,
}

STUB_SCRIPT = Path(__file__).parent / "mock_agent.py"


def main() -> None:
    procs: list[subprocess.Popen] = []

    print("[stubs] Launching mock agent stubs ...\n")
    for name, port in AGENTS.items():
        p = subprocess.Popen(
            [sys.executable, str(STUB_SCRIPT), name, str(port)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        procs.append(p)
        print(f"   [ok] {name}-agent   -> http://localhost:{port}  (PID {p.pid})")

    print(f"\n{'='*55}")
    print("  All stubs running. Press Ctrl+C to stop.\n")

    def _shutdown(sig, frame):
        print("\n[stubs] Stopping stubs ...")
        for p in procs:
            p.terminate()
        for p in procs:
            try:
                p.wait(timeout=5)
            except subprocess.TimeoutExpired:
                p.kill()
        print("   Done.")
        sys.exit(0)

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    # Keep parent alive and surface any stderr from children
    while True:
        time.sleep(1)
        for p in procs:
            if p.stderr:
                line = p.stderr.readline()
                if line:
                    print(line.decode().rstrip())


if __name__ == "__main__":
    main()
