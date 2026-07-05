"""API key management — enterprise programmatic access.

Keys look like: athr_<48 hex chars>
Only the SHA-256 hash is stored; the raw key is returned once on creation.
"""
from __future__ import annotations

import hashlib
import secrets
import time
import uuid
from typing import Optional

from orca.auth.db import get_conn, BACKEND


def _ensure_table() -> None:
    with get_conn() as conn:
        if BACKEND == "postgres":
            # Same concurrent-CREATE-TABLE race as orca/auth/db.py's init_db() —
            # serialize schema creation across instances so a simultaneous
            # multi-instance startup doesn't crash on a catalog UniqueViolation.
            conn.execute("SELECT pg_advisory_xact_lock(hashtext('orca_apikeys_init'))")
        conn.execute("""
        CREATE TABLE IF NOT EXISTS api_keys (
            id         TEXT PRIMARY KEY,
            user_id    TEXT NOT NULL,
            key_hash   TEXT NOT NULL UNIQUE,
            name       TEXT NOT NULL DEFAULT '',
            created_at REAL NOT NULL,
            last_used  REAL,
            revoked    INTEGER NOT NULL DEFAULT 0
        )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS ix_apikey_user ON api_keys(user_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS ix_apikey_hash ON api_keys(key_hash)")


_ensure_table()


def _hash_key(raw: str) -> str:
    return hashlib.sha256(raw.encode()).hexdigest()


def create_key(user_id: str, name: str = "") -> tuple[str, str]:
    """Create a new API key. Returns (key_id, raw_key). raw_key shown once."""
    raw = "athr_" + secrets.token_hex(24)
    kid = str(uuid.uuid4())
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO api_keys (id, user_id, key_hash, name, created_at) VALUES (?,?,?,?,?)",
            (kid, user_id, _hash_key(raw), name, time.time()),
        )
    return kid, raw


def list_keys(user_id: str) -> list[dict]:
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT id, name, created_at, last_used, revoked "
            "FROM api_keys WHERE user_id=? ORDER BY created_at DESC",
            (user_id,),
        ).fetchall()
    return [dict(r) for r in rows]


def revoke_key(key_id: str, user_id: str) -> bool:
    with get_conn() as conn:
        cur = conn.execute(
            "UPDATE api_keys SET revoked=1 WHERE id=? AND user_id=?",
            (key_id, user_id),
        )
    return cur.rowcount > 0


def verify_api_key(raw_key: str) -> Optional[str]:
    """Verify an API key string. Returns user_id if valid, None otherwise."""
    if not raw_key.startswith("athr_"):
        return None
    h = _hash_key(raw_key)
    with get_conn() as conn:
        row = conn.execute(
            "SELECT id, user_id FROM api_keys WHERE key_hash=? AND revoked=0",
            (h,),
        ).fetchone()
        if row:
            conn.execute(
                "UPDATE api_keys SET last_used=? WHERE id=?",
                (time.time(), row["id"]),
            )
            return row["user_id"]
    return None
