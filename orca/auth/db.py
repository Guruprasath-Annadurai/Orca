"""
Database backend for Orca auth — users, sessions, usage, API keys.

Dual-backend by design:
  - Default (no ORCA_DATABASE_URL set): SQLite at ~/.orca/auth.db. This is
    the actual product for most Orca users — a single local install with
    zero setup. Nothing changes for them.
  - Production mode (ORCA_DATABASE_URL set, e.g. postgresql://...): Postgres.
    Needed once you're running multiple API instances behind a load balancer —
    SQLite's file lock doesn't work across processes/machines.

Call sites (orca/audit.py, orca/auth/store.py, orca/auth/apikeys.py) use
get_conn() and never touch sqlite3/psycopg directly, so backend selection
here is transparent to the rest of the codebase. The _PGConnAdapter class
below exists to make a psycopg connection quack like a sqlite3.Connection —
same execute()/executescript()/fetchone()/fetchall()/dict-row/context-manager
surface — so no other file needs to change.
"""
from __future__ import annotations

import os
import re
import sqlite3
from pathlib import Path

from orca.config import ORCA_HOME

AUTH_DB = ORCA_HOME / "auth.db"

BACKEND = "postgres" if os.environ.get("ORCA_DATABASE_URL") else "sqlite"

_PLACEHOLDER_RE = re.compile(r"\?")

_SCHEMA_SQLITE = """
CREATE TABLE IF NOT EXISTS users (
    id                  TEXT PRIMARY KEY,
    email               TEXT UNIQUE NOT NULL,
    name                TEXT NOT NULL DEFAULT '',
    password_hash       TEXT NOT NULL,
    tier                TEXT NOT NULL DEFAULT 'free',
    role                TEXT NOT NULL DEFAULT 'member',
    created_at          TEXT NOT NULL,
    verified            INTEGER NOT NULL DEFAULT 0,
    stripe_customer_id  TEXT,
    totp_secret         TEXT,
    totp_enabled        INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS usage_daily (
    user_id    TEXT NOT NULL,
    date       TEXT NOT NULL,
    messages   INTEGER NOT NULL DEFAULT 0,
    ultra_runs INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (user_id, date)
);
"""

# Postgres gets the full audit_log hash-chain schema from day one — there's
# no "legacy pre-chain" Postgres install to migrate, unlike SQLite where
# existing local users may have an old-schema audit_log table.
_SCHEMA_POSTGRES = """
CREATE TABLE IF NOT EXISTS users (
    id            TEXT PRIMARY KEY,
    email               TEXT UNIQUE NOT NULL,
    name                TEXT NOT NULL DEFAULT '',
    password_hash       TEXT NOT NULL,
    tier                TEXT NOT NULL DEFAULT 'free',
    role                TEXT NOT NULL DEFAULT 'member',
    created_at          TEXT NOT NULL,
    verified            INTEGER NOT NULL DEFAULT 0,
    stripe_customer_id  TEXT,
    totp_secret         TEXT,
    totp_enabled        INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS ix_users_stripe_customer ON users(stripe_customer_id);

CREATE TABLE IF NOT EXISTS usage_daily (
    user_id    TEXT NOT NULL,
    date       TEXT NOT NULL,
    messages   INTEGER NOT NULL DEFAULT 0,
    ultra_runs INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (user_id, date)
);

CREATE TABLE IF NOT EXISTS audit_log (
    id          TEXT PRIMARY KEY,
    seq         BIGINT,
    user_id     TEXT,
    event       TEXT NOT NULL,
    detail      TEXT,
    ip          TEXT,
    created_at  DOUBLE PRECISION NOT NULL,
    prev_hash   TEXT NOT NULL,
    entry_hash  TEXT NOT NULL,
    signature   TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_audit_user  ON audit_log(user_id);
CREATE INDEX IF NOT EXISTS ix_audit_event ON audit_log(event);
CREATE INDEX IF NOT EXISTS ix_audit_ts    ON audit_log(created_at);
CREATE INDEX IF NOT EXISTS ix_audit_seq   ON audit_log(seq);
"""
# api_keys is NOT created here — orca/auth/apikeys.py owns that table's
# schema (it has its own _ensure_table() with REAL timestamp columns and a
# `revoked` flag this module must not shadow). session_titles and doc_registry
# stay as flat JSON files for now — out of scope for this migration pass.


class _PGCursorAdapter:
    """Wraps a psycopg cursor so callers can use it exactly like a sqlite3 cursor."""

    def __init__(self, cur):
        self._cur = cur

    def fetchone(self):
        return self._cur.fetchone()

    def fetchall(self):
        return self._cur.fetchall()

    @property
    def rowcount(self):
        return self._cur.rowcount


class _PGConnAdapter:
    """
    Makes a psycopg connection quack like sqlite3.Connection:
      - execute(sql, params) with '?' placeholders (translated to '%s')
      - executescript(sql) for multi-statement DDL blocks
      - dict-row access (row["col"]) via psycopg's dict_row factory
      - context-manager commit-on-success / rollback-on-exception

    Deliberately does NOT close the underlying connection on __exit__ —
    this matches the existing sqlite3 usage pattern in this codebase (every
    call site does `with get_conn() as conn:` and relies on commit-only
    semantics, never calling conn.close()). Not fixing that pattern here;
    out of scope for this migration.
    """

    def __init__(self, pg_conn):
        self._conn = pg_conn

    def execute(self, sql: str, params=()):
        sql = _PLACEHOLDER_RE.sub("%s", sql)
        cur = self._conn.cursor()
        cur.execute(sql, params)
        return _PGCursorAdapter(cur)

    def executescript(self, script: str) -> None:
        cur = self._conn.cursor()
        for stmt in filter(None, (s.strip() for s in script.split(";"))):
            cur.execute(stmt)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        if exc_type is None:
            self._conn.commit()
        else:
            self._conn.rollback()
        return False  # never suppress exceptions


def _get_postgres_conn() -> _PGConnAdapter:
    import psycopg
    from psycopg.rows import dict_row

    dsn = os.environ["ORCA_DATABASE_URL"]
    conn = psycopg.connect(dsn, row_factory=dict_row, autocommit=False)
    return _PGConnAdapter(conn)


def _get_sqlite_conn() -> sqlite3.Connection:
    AUTH_DB.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(AUTH_DB), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def get_conn():
    """Returns a sqlite3.Connection or a _PGConnAdapter, selected by ORCA_DATABASE_URL."""
    if BACKEND == "postgres":
        return _get_postgres_conn()
    return _get_sqlite_conn()


def init_db() -> None:
    if BACKEND == "postgres":
        with get_conn() as conn:
            # CREATE TABLE IF NOT EXISTS is NOT safe under concurrent creation
            # on Postgres — multiple instances racing to create the same table
            # at startup can hit a catalog-level UniqueViolation
            # (pg_type_typname_nsp_index) even with IF NOT EXISTS, because the
            # check-then-create isn't atomic across transactions. Every API
            # replica runs this at import time, so without serializing it here,
            # a simultaneous multi-instance rollout crashes on boot. Same
            # advisory-lock pattern as the audit chain writer: first instance
            # through does the real work, the rest block, then see the schema
            # already exists once they get the lock.
            conn.execute("SELECT pg_advisory_xact_lock(hashtext('orca_schema_init'))")
            conn.executescript(_SCHEMA_POSTGRES)
            # A Postgres deployment created BEFORE a given column existed in
            # _SCHEMA_POSTGRES needs its own migration too — "fresh installs
            # get every column" only covers installs created after this line
            # was added. Postgres supports ADD COLUMN IF NOT EXISTS natively,
            # no try/except dance needed like the SQLite branch below.
            conn.execute("ALTER TABLE users ADD COLUMN IF NOT EXISTS stripe_customer_id TEXT")
            conn.execute("CREATE INDEX IF NOT EXISTS ix_users_stripe_customer ON users(stripe_customer_id)")
            conn.execute("ALTER TABLE users ADD COLUMN IF NOT EXISTS totp_secret TEXT")
            conn.execute("ALTER TABLE users ADD COLUMN IF NOT EXISTS totp_enabled INTEGER NOT NULL DEFAULT 0")
        return

    with get_conn() as conn:
        conn.executescript(_SCHEMA_SQLITE)
        # Migrations for existing SQLite installs only — a fresh Postgres
        # schema above already has every column (as of its own creation date —
        # columns added after that still need the ALTER above).
        for stmt in [
            "ALTER TABLE users ADD COLUMN role TEXT NOT NULL DEFAULT 'member'",
            "ALTER TABLE users ADD COLUMN stripe_customer_id TEXT",
            "ALTER TABLE users ADD COLUMN totp_secret TEXT",
            "ALTER TABLE users ADD COLUMN totp_enabled INTEGER NOT NULL DEFAULT 0",
        ]:
            try:
                conn.execute(stmt)
            except Exception:
                pass  # column already exists


init_db()
