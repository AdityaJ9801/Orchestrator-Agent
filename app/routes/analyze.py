"""
FastAPI router: POST /analyze  (full execution, returns JSON response)
"""
from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException

from app.config import Settings, get_settings
from app.executor import execute_graph
from app.models import AnalyzeRequest, AnalyzeResponse, TaskStatus
from app.planner import plan_query
from app.session_store import load_session, new_session_id, save_session

logger = logging.getLogger(__name__)
router = APIRouter(tags=["analyze"])


@router.post("/analyze", response_model=AnalyzeResponse, summary="Analyze a user query")
async def analyze(
    request: AnalyzeRequest,
    settings: Settings = Depends(get_settings),
) -> AnalyzeResponse:
    """
    Full pipeline:
    1. Plan → TaskGraph (via LLM)
    2. Execute graph (parallel where possible)
    3. Persist session to Redis
    4. Return unified response
    """
    task_id    = str(uuid.uuid4())
    session_id = request.session_id or new_session_id()

    logger.info(
        "analyze start task_id=%s session_id=%s query=%r context_id=%s",
        task_id, session_id, request.query[:80], request.context_id,
    )

    # 1. Plan
    try:
        graph = await plan_query(request.query, request.context_id)
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=str(exc))

    # 2. Execute
    results = await execute_graph(graph)

    partial = any(
        r.status in (TaskStatus.FAILED, TaskStatus.SKIPPED) for r in results
    )

    response = AnalyzeResponse(
        task_id=task_id,
        session_id=session_id,
        intent=graph.intent,
        graph=graph,
        results=results,
        partial=partial,
    )

    # 3. Persist
    try:
        await save_session(session_id, response.model_dump())
    except Exception as exc:
        logger.warning("Redis session save failed (non-fatal): %s", exc)

    return response
