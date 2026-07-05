"""
Redis-backed session continuity — makes conversation history survive process
restarts and cross-instance load-balanced traffic.

Problem: the `_sessions` dict in orca/serve/api.py is process-local. Two
genuinely-volatile pieces of state live only in that process's memory:
  - AgentLoop._history (the turn-by-turn conversation the model sees)
  - which model_variant this session is pinned to

Everything else a session touches is already durable and shared, so it needs
no Redis involvement:
  - DocStore (RAG) — ChromaDB + JSON registry on disk
  - EpisodicMemory / LongTermMemory — JSON files / ChromaDB on disk
  - ExplainStore — deliberately ephemeral (documented in brain/explainability.py),
    low-stakes if lost, not worth the complexity of replicating

If a load-balanced request for an existing session lands on an API instance
that doesn't have it in local memory (process restart, different replica),
today's code silently starts a fresh AgentLoop with empty history — the user
just loses conversation context with no error. This module fixes that.

Opt-in via ORCA_REDIS_URL, same pattern as ORCA_DATABASE_URL for Postgres —
unset means exactly today's in-memory-only behavior (single instance, local
dev, zero setup required). Every function here is a no-op when disabled and
never raises when Redis is unreachable — losing session continuity is bad,
but crashing the chat request over it would be worse.
"""
from __future__ import annotations

import json
import os
import time
from typing import Optional

REDIS_URL = os.environ.get("ORCA_REDIS_URL")
SESSION_TTL_SECONDS = 7200  # matches the existing 2h idle-eviction window in api.py

_redis_client = None


def enabled() -> bool:
    return REDIS_URL is not None


def _client():
    global _redis_client
    if _redis_client is None:
        import redis
        _redis_client = redis.from_url(REDIS_URL, decode_responses=True)
    return _redis_client


def _key(session_id: str) -> str:
    return f"orca:session:{session_id}"


def save_session_state(session_id: str, model_variant: str, history: list[dict]) -> None:
    """Persist just enough to reconstruct this session on any instance."""
    if not enabled():
        return
    try:
        payload = json.dumps({
            "model_variant": model_variant,
            "history": history,
            "saved_at": time.time(),
        })
        _client().set(_key(session_id), payload, ex=SESSION_TTL_SECONDS)
    except Exception:
        pass  # Redis being down must never break the chat request


def load_session_state(session_id: str) -> Optional[dict]:
    """Returns {'model_variant': str, 'history': list[dict], 'saved_at': float} or None."""
    if not enabled():
        return None
    try:
        raw = _client().get(_key(session_id))
        if not raw:
            return None
        return json.loads(raw)
    except Exception:
        return None


def touch_session(session_id: str) -> None:
    """Refresh TTL without rewriting the payload — cheap keepalive per request."""
    if not enabled():
        return
    try:
        _client().expire(_key(session_id), SESSION_TTL_SECONDS)
    except Exception:
        pass


def delete_session(session_id: str) -> None:
    if not enabled():
        return
    try:
        _client().delete(_key(session_id))
    except Exception:
        pass


def ping() -> bool:
    """Health check — used by /api/status to report Redis connectivity honestly."""
    if not enabled():
        return False
    try:
        return bool(_client().ping())
    except Exception:
        return False
