"""
FastAPI router: POST /plan  (dry-run — returns TaskGraph without calling agents)
"""
from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException

from app.config import Settings, get_settings
from app.models import PlanRequest, TaskGraph
from app.planner import plan_query

logger = logging.getLogger(__name__)
router = APIRouter(tags=["plan"])


@router.post("/plan", response_model=TaskGraph, summary="Dry-run: return task graph only")
async def plan(
    request: PlanRequest,
    settings: Settings = Depends(get_settings),
) -> TaskGraph:
    """
    Ask the LLM to decompose the query into a TaskGraph.
    No agents are called; useful for inspection and debugging.
    """
    try:
        graph = await plan_query(request.query, request.context_id)
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=str(exc))
    return graph
