"""
Mock Agent Stub
===============
A single FastAPI app that pretends to be ONE specialist agent.
Launch 6 instances on ports 8001-8006 via run_stubs.py.

Each stub:
  GET  /health  → {"status":"ok","agent":<name>}
  POST /run     → returns a canned realistic-looking result for that agent type
"""
from __future__ import annotations

import random
import sys
import time
from datetime import datetime, timezone

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

AGENT_RESPONSES = {
    "context": lambda body: {
        "context_id":   body.get("context_id", "ctx-demo"),
        "dataset_name": "sales_2024",
        "row_count":    142_500,
        "columns": ["date", "region", "product", "revenue", "units"],
        "date_range":   {"from": "2024-01-01", "to": "2024-12-31"},
        "freshness_ms": random.randint(10, 80),
    },
    "sql": lambda body: {
        "query_executed": "SELECT region, SUM(revenue) AS total FROM sales_2024 GROUP BY region",
        "rows": [
            {"region": "North", "total": 1_840_200},
            {"region": "South", "total": 2_310_450},
            {"region": "East",  "total": 990_780},
            {"region": "West",  "total": 1_530_000},
        ],
        "duration_ms": random.randint(30, 200),
    },
    "viz": lambda body: {
        "chart_type":   "bar",
        "title":        "Revenue by Region – 2024",
        "format":       "svg",
        "data_points":  4,
        "render_ms":    random.randint(20, 90),
        "url":          "/charts/revenue_by_region_2024.svg",
    },
    "ml": lambda body: {
        "model":        "XGBRegressor",
        "target":       "revenue",
        "r2_score":     round(random.uniform(0.81, 0.96), 4),
        "rmse":         round(random.uniform(4200, 9800), 2),
        "top_features": ["region", "product", "month"],
        "trained_on":   142_500,
    },
    "nlp": lambda body: {
        "sentiment":    "positive",
        "score":        round(random.uniform(0.6, 0.95), 3),
        "topics":       ["revenue growth", "regional performance"],
        "entities":     ["Q4 2024", "North region"],
        "summary":      "Sales showed strong growth in the Southern region with positive overall sentiment.",
    },
    "report": lambda body: {
        "report_id":    f"rpt-{int(time.time())}",
        "format":       "pdf",
        "pages":        5,
        "sections":     ["executive_summary", "data_analysis", "visualizations", "ml_insights", "recommendations"],
        "download_url": "/reports/analysis_demo.pdf",
        "generated_at": datetime.now(tz=timezone.utc).isoformat(),
    },
}

VALID_AGENTS = list(AGENT_RESPONSES.keys())


def create_stub_app(agent_name: str) -> FastAPI:
    if agent_name not in VALID_AGENTS:
        raise ValueError(f"Unknown agent: {agent_name}. Choose from {VALID_AGENTS}")

    app = FastAPI(title=f"Mock {agent_name.capitalize()} Agent", docs_url="/docs")
    handler = AGENT_RESPONSES[agent_name]

    @app.get("/health")
    async def health():
        return {"status": "ok", "agent": agent_name, "mode": "stub"}

    @app.post("/run")
    async def run(request: Request):
        body = await request.json()
        # Simulate a small realistic delay
        await __import__("asyncio").sleep(random.uniform(0.05, 0.25))
        return JSONResponse(content={
            "agent":      agent_name,
            "task_id":    body.get("task_id", "unknown"),
            "result":     handler(body),
            "stub":       True,
        })

    return app


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python mock_agent.py <agent_name> <port>")
        print(f"  agent_name: one of {VALID_AGENTS}")
        sys.exit(1)

    name = sys.argv[1]
    port = int(sys.argv[2])
    stub = create_stub_app(name)
    print(f"[stub] Starting mock {name}-agent on port {port}")
    uvicorn.run(stub, host="0.0.0.0", port=port, log_level="warning")
