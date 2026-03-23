"""
Redis-backed session store.

Each session is keyed as  orchestrator:session:<session_id>
Value is a JSON-serialised dict containing the AnalyzeResponse payload.
TTL defaults to 1 h (from settings.session_ttl_seconds).
"""
from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime
from typing import Any, Dict, Optional

import redis.asyncio as aioredis

from app.config import get_settings

logger = logging.getLogger(__name__)

_PREFIX = "orchestrator:session:"


def _make_key(session_id: str) -> str:
    return f"{_PREFIX}{session_id}"


def _new_session_id() -> str:
    return str(uuid.uuid4())


# ── Redis connection pool (module-level singleton) ────────────────────────────

_redis: Optional[aioredis.Redis] = None


async def get_redis() -> aioredis.Redis:
    global _redis
    if _redis is None:
        settings = get_settings()
        _redis = aioredis.from_url(
            settings.redis_url,
            encoding="utf-8",
            decode_responses=True,
        )
    return _redis


async def close_redis() -> None:
    global _redis
    if _redis is not None:
        await _redis.aclose()
        _redis = None


# ── Public API ────────────────────────────────────────────────────────────────

async def save_session(session_id: str, data: Dict[str, Any]) -> None:
    """Persist session data to Redis with configured TTL."""
    settings = get_settings()
    r = await get_redis()
    serialised = json.dumps(data, default=str)
    await r.setex(_make_key(session_id), settings.session_ttl_seconds, serialised)
    logger.debug("Session %s saved (%d bytes)", session_id, len(serialised))


async def load_session(session_id: str) -> Optional[Dict[str, Any]]:
    """Load session from Redis; returns None if missing/expired."""
    r = await get_redis()
    raw = await r.get(_make_key(session_id))
    if raw is None:
        return None
    return json.loads(raw)


async def delete_session(session_id: str) -> None:
    r = await get_redis()
    await r.delete(_make_key(session_id))


def new_session_id() -> str:
    return _new_session_id()
