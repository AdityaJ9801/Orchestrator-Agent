"""
Tests for Redis session store (app/session_store.py).

Scenarios:
  1.  save_session stores data and load_session retrieves it
  2.  load_session returns None for missing key
  3.  delete_session removes the key
  4.  new_session_id returns a valid UUID string
  5.  new_session_id returns unique values on each call
  6.  save_session serialises datetime fields without error
"""
from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.session_store import (
    delete_session,
    load_session,
    new_session_id,
    save_session,
)


# ── Redis mock ────────────────────────────────────────────────────────────────

def _mock_redis():
    """Return a mock Redis client with async setex/get/delete."""
    r = AsyncMock()
    _store: dict[str, str] = {}

    async def _setex(key, ttl, value):
        _store[key] = value

    async def _get(key):
        return _store.get(key)

    async def _delete(key):
        _store.pop(key, None)

    r.setex  = AsyncMock(side_effect=_setex)
    r.get    = AsyncMock(side_effect=_get)
    r.delete = AsyncMock(side_effect=_delete)
    return r


# ── Tests ─────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_save_and_load_session():
    mock_r = _mock_redis()
    with patch("app.session_store.get_redis", new=AsyncMock(return_value=mock_r)):
        await save_session("sess-1", {"intent": "test", "results": []})
        data = await load_session("sess-1")

    assert data is not None
    assert data["intent"] == "test"


@pytest.mark.asyncio
async def test_load_session_missing_returns_none():
    mock_r = _mock_redis()
    with patch("app.session_store.get_redis", new=AsyncMock(return_value=mock_r)):
        data = await load_session("nonexistent-session")

    assert data is None


@pytest.mark.asyncio
async def test_delete_session():
    mock_r = _mock_redis()
    with patch("app.session_store.get_redis", new=AsyncMock(return_value=mock_r)):
        await save_session("sess-del", {"key": "value"})
        await delete_session("sess-del")
        data = await load_session("sess-del")

    assert data is None


def test_new_session_id_is_valid_uuid():
    sid = new_session_id()
    # Should not raise
    uuid.UUID(sid)


def test_new_session_id_unique():
    ids = {new_session_id() for _ in range(10)}
    assert len(ids) == 10


@pytest.mark.asyncio
async def test_save_session_with_datetime_fields():
    """Datetime values should be serialised without raising TypeError."""
    mock_r = _mock_redis()
    data = {
        "created_at": datetime.now(tz=timezone.utc),
        "intent": "test",
    }
    with patch("app.session_store.get_redis", new=AsyncMock(return_value=mock_r)):
        # Should not raise
        await save_session("sess-dt", data)
        loaded = await load_session("sess-dt")

    assert loaded is not None
    assert loaded["intent"] == "test"
