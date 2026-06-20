"""Audit log — every significant server event stored to SQLite."""
from __future__ import annotations

import json
import time
import uuid

from orca.auth.db import get_conn


def _ensure_table() -> None:
    with get_conn() as conn:
        conn.execute("""
        CREATE TABLE IF NOT EXISTS audit_log (
            id         TEXT PRIMARY KEY,
            user_id    TEXT,
            event      TEXT NOT NULL,
            detail     TEXT,
            ip         TEXT,
            created_at REAL NOT NULL
        )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS ix_audit_user  ON audit_log(user_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS ix_audit_event ON audit_log(event)")
        conn.execute("CREATE INDEX IF NOT EXISTS ix_audit_ts    ON audit_log(created_at)")


_ensure_table()


def log(
    event: str,
    user_id: str | None = None,
    detail: dict | None = None,
    ip: str | None = None,
) -> None:
    """Fire-and-forget audit entry. Never raises."""
    try:
        with get_conn() as conn:
            conn.execute(
                "INSERT INTO audit_log (id, user_id, event, detail, ip, created_at) "
                "VALUES (?,?,?,?,?,?)",
                (
                    str(uuid.uuid4()),
                    user_id,
                    event,
                    json.dumps(detail) if detail else None,
                    ip,
                    time.time(),
                ),
            )
    except Exception:
        pass


def recent(limit: int = 100, user_id: str | None = None) -> list[dict]:
    with get_conn() as conn:
        if user_id:
            rows = conn.execute(
                "SELECT * FROM audit_log WHERE user_id=? ORDER BY created_at DESC LIMIT ?",
                (user_id, limit),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM audit_log ORDER BY created_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
    return [
        {
            "id":         r["id"],
            "user_id":    r["user_id"],
            "event":      r["event"],
            "detail":     json.loads(r["detail"]) if r["detail"] else None,
            "ip":         r["ip"],
            "created_at": r["created_at"],
        }
        for r in rows
    ]
