"""
FastAPI router: POST /analyze/stream  →  Server-Sent Events
"""
from __future__ import annotations

import asyncio
import json
import logging
import uuid
from datetime import datetime, timezone
from typing import AsyncGenerator

from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse

from app.config import Settings, get_settings
from app.executor import execute_graph
from app.models import AnalyzeRequest, SSEEvent, TaskResult, TaskStatus
from app.planner import plan_query
from app.session_store import new_session_id, save_session

logger = logging.getLogger(__name__)
router = APIRouter(tags=["stream"])

# SSE helpers ─────────────────────────────────────────────────────────────────

def _sse(event: str, data: dict) -> str:
    payload = json.dumps(data, default=str)
    return f"event: {event}\ndata: {payload}\n\n"


# ── Streaming generator ───────────────────────────────────────────────────────

async def _stream_analysis(
    request: AnalyzeRequest,
    task_id: str,
    session_id: str,
    queue: asyncio.Queue,
) -> None:
    """Run inside a task; pushes SSE strings into *queue*. Sentinel = None."""
    try:
        graph = await plan_query(request.query, request.context_id)
        await queue.put(_sse("plan", {"task_id": task_id, "intent": graph.intent, "graph": graph.model_dump()}))

        async def on_start(r: TaskResult) -> None:
            await queue.put(_sse("task_start", {
                "task_id":    task_id,
                "node_id":    r.task_id,
                "agent":      r.agent.value,
                "started_at": r.started_at.isoformat() if r.started_at else None,
            }))

        async def on_done(r: TaskResult) -> None:
            await queue.put(_sse("task_complete", {
                "task_id":  task_id,
                "node_id":  r.task_id,
                "agent":    r.agent.value,
                "status":   r.status.value,
                "duration_ms": r.duration_ms,
                "error":    r.error,
            }))

        # Pass file attachments so every agent can access uploaded files (multimodal)
        file_dicts = [f.model_dump() for f in request.files] if request.files else None
        results = await execute_graph(
            graph,
            on_task_start=on_start,
            on_task_done=on_done,
            files=file_dicts,
        )

        partial = any(
            r.status in (TaskStatus.FAILED, TaskStatus.SKIPPED) for r in results
        )

        final_payload = {
            "task_id":    task_id,
            "session_id": session_id,
            "intent":     graph.intent,
            "graph":      graph.model_dump(),
            "results":    [r.model_dump() for r in results],
            "partial":    partial,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        await queue.put(_sse("result", final_payload))

        # Persist session
        try:
            await save_session(session_id, final_payload)
        except Exception as exc:
            logger.warning("Redis save failed: %s", exc)

    except Exception as exc:
        logger.exception("Stream analysis error")
        await queue.put(_sse("error", {"task_id": task_id, "detail": str(exc)}))
    finally:
        await queue.put(None)  # sentinel


async def _event_generator(
    request: AnalyzeRequest,
    task_id: str,
    session_id: str,
) -> AsyncGenerator[str, None]:
    queue: asyncio.Queue = asyncio.Queue()
    asyncio.create_task(_stream_analysis(request, task_id, session_id, queue))

    while True:
        item = await queue.get()
        if item is None:
            break
        yield item


# ── Route ─────────────────────────────────────────────────────────────────────

@router.post("/analyze/stream", summary="Stream analysis via SSE")
async def analyze_stream(
    body: AnalyzeRequest,
    settings: Settings = Depends(get_settings),
) -> StreamingResponse:
    """
    Returns Server-Sent Events:
      • event: plan          — task graph ready
      • event: task_start    — individual agent task started
      • event: task_complete — individual agent task finished
      • event: result        — final assembled result
      • event: error         — unrecoverable error
    """
    task_id    = str(uuid.uuid4())
    session_id = body.session_id or new_session_id()

    return StreamingResponse(
        _event_generator(body, task_id, session_id),
        media_type="text/event-stream",
        headers={
            "Cache-Control":    "no-cache",
            "X-Accel-Buffering": "no",
        },
    )
