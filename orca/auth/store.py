"""User CRUD and daily usage tracking."""
from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import date, datetime
from typing import Optional

from orca.auth.db import get_conn
from orca.auth.crypto import hash_password, verify_password

DAILY_LIMITS: dict[str, dict[str, int]] = {
    "free":       {"messages": 50,  "ultra": 3},
    "pro":        {"messages": -1,  "ultra": 50},
    "enterprise": {"messages": -1,  "ultra": -1},
}


@dataclass
class User:
    id: str
    email: str
    name: str
    tier: str
    verified: bool
    role: str = "member"


# ── User CRUD ─────────────────────────────────────────────────────────────────

def mark_verified(user_id: str) -> None:
    with get_conn() as conn:
        conn.execute("UPDATE users SET verified=1 WHERE id=?", (user_id,))


def update_password(user_id: str, new_password: str) -> None:
    from orca.auth.crypto import hash_password
    with get_conn() as conn:
        conn.execute("UPDATE users SET password_hash=? WHERE id=?", (hash_password(new_password), user_id))


def _count_users() -> int:
    with get_conn() as conn:
        row = conn.execute("SELECT COUNT(*) as n FROM users").fetchone()
    return row["n"] if row else 0


def create_user(email: str, password: str, name: str = "") -> User:
    uid = str(uuid.uuid4())
    ph  = hash_password(password)
    now = datetime.utcnow().isoformat()
    display = name or email.split("@")[0]
    role = "owner" if _count_users() == 0 else "member"
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO users (id, email, name, password_hash, tier, role, created_at) "
            "VALUES (?,?,?,?,?,?,?)",
            (uid, email.lower().strip(), display, ph, "free", role, now),
        )
    return User(id=uid, email=email, name=display, tier="free", verified=False, role=role)


def set_user_tier(user_id: str, tier: str) -> None:
    with get_conn() as conn:
        conn.execute("UPDATE users SET tier=? WHERE id=?", (tier, user_id))


def set_stripe_customer_id(user_id: str, customer_id: str) -> None:
    with get_conn() as conn:
        conn.execute("UPDATE users SET stripe_customer_id=? WHERE id=?", (customer_id, user_id))


def get_user_by_stripe_customer_id(customer_id: str) -> Optional[User]:
    with get_conn() as conn:
        row = conn.execute(
            "SELECT * FROM users WHERE stripe_customer_id=?", (customer_id,)
        ).fetchone()
    return _row_to_user(row)


def get_stripe_customer_id(user_id: str) -> Optional[str]:
    """stripe_customer_id isn't on the User dataclass (billing-internal detail) — fetch directly."""
    with get_conn() as conn:
        row = conn.execute(
            "SELECT stripe_customer_id FROM users WHERE id=?", (user_id,)
        ).fetchone()
    return row["stripe_customer_id"] if row else None


# ── 2FA / TOTP ─────────────────────────────────────────────────────────────────

def set_pending_totp_secret(user_id: str, secret: str) -> None:
    """
    Stores a secret WITHOUT enabling 2FA — the user must prove they can
    generate a valid code (via enable_totp) before it takes effect. Prevents
    a half-finished setup from silently requiring 2FA on next login with no
    way for the user to have actually saved the secret in their app.
    """
    with get_conn() as conn:
        conn.execute("UPDATE users SET totp_secret=?, totp_enabled=0 WHERE id=?", (secret, user_id))


def enable_totp(user_id: str) -> None:
    with get_conn() as conn:
        conn.execute("UPDATE users SET totp_enabled=1 WHERE id=?", (user_id,))


def disable_totp(user_id: str) -> None:
    with get_conn() as conn:
        conn.execute("UPDATE users SET totp_secret=NULL, totp_enabled=0 WHERE id=?", (user_id,))


def get_totp_state(user_id: str) -> dict:
    """Returns {'secret': str|None, 'enabled': bool} — internal detail, not on the User dataclass."""
    with get_conn() as conn:
        row = conn.execute(
            "SELECT totp_secret, totp_enabled FROM users WHERE id=?", (user_id,)
        ).fetchone()
    if not row:
        return {"secret": None, "enabled": False}
    return {"secret": row["totp_secret"], "enabled": bool(row["totp_enabled"])}


def set_user_role(user_id: str, role: str) -> None:
    with get_conn() as conn:
        conn.execute("UPDATE users SET role=? WHERE id=?", (role, user_id))


def list_users(limit: int = 100, offset: int = 0) -> list[dict]:
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT id, email, name, tier, role, created_at, verified "
            "FROM users ORDER BY created_at DESC LIMIT ? OFFSET ?",
            (limit, offset),
        ).fetchall()
    return [dict(r) for r in rows]


def get_user_by_email(email: str) -> Optional[User]:
    with get_conn() as conn:
        row = conn.execute(
            "SELECT * FROM users WHERE email=?", (email.lower().strip(),)
        ).fetchone()
    return _row_to_user(row)


def get_user_by_id(uid: str) -> Optional[User]:
    with get_conn() as conn:
        row = conn.execute("SELECT * FROM users WHERE id=?", (uid,)).fetchone()
    return _row_to_user(row)


def authenticate(email: str, password: str) -> Optional[User]:
    with get_conn() as conn:
        row = conn.execute(
            "SELECT * FROM users WHERE email=?", (email.lower().strip(),)
        ).fetchone()
    if not row or not verify_password(password, row["password_hash"]):
        return None
    return _row_to_user(row)


def upgrade_tier(user_id: str, tier: str) -> None:
    with get_conn() as conn:
        conn.execute("UPDATE users SET tier=? WHERE id=?", (tier, user_id))




def _row_to_user(row) -> Optional[User]:
    if not row:
        return None
    cols = row.keys() if hasattr(row, "keys") else []
    return User(
        id=row["id"],
        email=row["email"],
        name=row["name"],
        tier=row["tier"],
        verified=bool(row["verified"]),
        role=row["role"] if "role" in cols else "member",
    )


# ── Usage / quota ─────────────────────────────────────────────────────────────

def check_quota(user_id: str, tier: str, kind: str = "message") -> tuple[bool, int, int]:
    """Returns (allowed, used_today, daily_limit). -1 limit means unlimited."""
    today = date.today().isoformat()
    col   = "messages" if kind == "message" else "ultra_runs"
    limit = DAILY_LIMITS.get(tier, DAILY_LIMITS["free"]).get(
        "messages" if kind == "message" else "ultra", 50
    )
    with get_conn() as conn:
        row = conn.execute(
            f"SELECT {col} FROM usage_daily WHERE user_id=? AND date=?",
            (user_id, today),
        ).fetchone()
    used = row[col] if row else 0
    if limit == -1:
        return True, used, limit
    return used < limit, used, limit


def increment_usage(user_id: str, kind: str = "message") -> None:
    today = date.today().isoformat()
    col   = "messages" if kind == "message" else "ultra_runs"
    with get_conn() as conn:
        conn.execute(
            f"INSERT INTO usage_daily (user_id, date, {col}) VALUES (?,?,1) "
            f"ON CONFLICT(user_id, date) DO UPDATE SET {col}={col}+1",
            (user_id, today),
        )


def get_usage_today(user_id: str) -> dict:
    today = date.today().isoformat()
    with get_conn() as conn:
        row = conn.execute(
            "SELECT * FROM usage_daily WHERE user_id=? AND date=?", (user_id, today)
        ).fetchone()
    if not row:
        return {"messages": 0, "ultra_runs": 0}
    return {"messages": row["messages"], "ultra_runs": row["ultra_runs"]}


# ── User <-> session mapping (needed for account deletion to actually cascade) ──
# Chat history, uploaded documents, and memory files are keyed purely by
# session_id — nothing tied them to an owning user_id before this table
# existed, meaning "delete my account" had no way to find and remove them.

def record_user_session(user_id: str, session_id: str) -> None:
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO user_sessions (user_id, session_id, created_at) VALUES (?,?,?) "
            "ON CONFLICT(user_id, session_id) DO NOTHING",
            (user_id, session_id, datetime.utcnow().isoformat()),
        )


def get_user_session_ids(user_id: str) -> list[str]:
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT session_id FROM user_sessions WHERE user_id=?", (user_id,)
        ).fetchall()
    return [r["session_id"] for r in rows]


def delete_user_account_records(user_id: str) -> None:
    """Deletes the account-level rows (users, usage_daily, api_keys, user_sessions).
    Does NOT touch session-scoped data (memory files, docs, Redis) — that's
    orchestrated separately in orca/serve/account_delete.py, since it needs
    access to DocStore/EpisodicMemory/session_store, which this module
    intentionally doesn't import (keeps store.py's dependency footprint to
    just the DB layer)."""
    with get_conn() as conn:
        conn.execute("DELETE FROM api_keys WHERE user_id=?", (user_id,))
        conn.execute("DELETE FROM usage_daily WHERE user_id=?", (user_id,))
        conn.execute("DELETE FROM user_sessions WHERE user_id=?", (user_id,))
        conn.execute("DELETE FROM users WHERE id=?", (user_id,))
