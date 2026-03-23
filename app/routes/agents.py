"""
FastAPI router: GET /agents/status  — health-check all downstream agents in parallel
"""
from __future__ import annotations

import asyncio
import logging
import time
from typing import List

import httpx
from fastapi import APIRouter, Depends

from app.config import Settings, get_settings
from app.models import AgentHealthInfo, AgentsStatusResponse

logger = logging.getLogger(__name__)
router = APIRouter(tags=["agents"])

_HEALTH_TIMEOUT = 5.0  # seconds per agent


async def _check_one(
    name: str,
    url: str,
    client: httpx.AsyncClient,
) -> AgentHealthInfo:
    start = time.monotonic()
    try:
        resp = await client.get(f"{url}/health", timeout=_HEALTH_TIMEOUT)
        latency = (time.monotonic() - start) * 1000
        return AgentHealthInfo(
            agent=name,
            url=url,
            healthy=resp.status_code == 200,
            latency_ms=round(latency, 2),
        )
    except Exception as exc:
        latency = (time.monotonic() - start) * 1000
        return AgentHealthInfo(
            agent=name,
            url=url,
            healthy=False,
            latency_ms=round(latency, 2),
            error=str(exc),
        )


@router.get(
    "/agents/status",
    response_model=AgentsStatusResponse,
    summary="Health-check all downstream agents",
)
async def agents_status(
    settings: Settings = Depends(get_settings),
) -> AgentsStatusResponse:
    """
    Calls /health on every registered agent in parallel.
    Returns aggregated health status.
    """
    registry = settings.agent_registry

    async with httpx.AsyncClient() as client:
        checks = await asyncio.gather(
            *[_check_one(name, url, client) for name, url in registry.items()],
            return_exceptions=False,
        )

    return AgentsStatusResponse(agents=list(checks))
