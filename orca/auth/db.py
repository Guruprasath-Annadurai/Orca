"""SQLite database setup for Atheris auth — users, sessions, usage."""
from __future__ import annotations

import sqlite3
from pathlib import Path

from orca.config import ORCA_HOME

AUTH_DB = ORCA_HOME / "auth.db"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id            TEXT PRIMARY KEY,
    email         TEXT UNIQUE NOT NULL,
    name          TEXT NOT NULL DEFAULT '',
    password_hash TEXT NOT NULL,
    tier          TEXT NOT NULL DEFAULT 'free',
    role          TEXT NOT NULL DEFAULT 'member',
    created_at    TEXT NOT NULL,
    verified      INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS usage_daily (
    user_id    TEXT NOT NULL,
    date       TEXT NOT NULL,
    messages   INTEGER NOT NULL DEFAULT 0,
    ultra_runs INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (user_id, date)
);
"""


def get_conn() -> sqlite3.Connection:
    AUTH_DB.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(AUTH_DB), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_db() -> None:
    with get_conn() as conn:
        conn.executescript(_SCHEMA)
        # Migrations for existing installs
        for stmt in [
            "ALTER TABLE users ADD COLUMN role TEXT NOT NULL DEFAULT 'member'",
        ]:
            try:
                conn.execute(stmt)
            except Exception:
                pass  # column already exists


init_db()
