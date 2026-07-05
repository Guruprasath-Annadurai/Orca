"""
One-time data migration: local SQLite (~/.orca/auth.db) → Postgres.

Run this once when cutting over a local/single-instance Orca deployment to
production Postgres. Copies users, usage_daily, api_keys, and audit_log rows
byte-for-byte — audit_log hash-chain values (entry_hash/signature/prev_hash)
are copied AS-IS, never recomputed, so the existing chain's cryptographic
history stays intact and verifiable after the move.

Usage:
    ORCA_DATABASE_URL=postgresql://user@host/dbname \
        uv run python3 -m orca.auth.migrate_to_postgres

Safe to re-run: every INSERT uses ON CONFLICT DO NOTHING, so partial re-runs
after a failure won't duplicate rows. Does NOT delete the source SQLite file —
verify the migration (this script runs verify_chain() at the end) before
tearing down the old database.
"""
from __future__ import annotations

import os
import sqlite3
import sys

from orca.config import ORCA_HOME


def _sqlite_source_conn() -> sqlite3.Connection:
    path = ORCA_HOME / "auth.db"
    if not path.exists():
        print(f"No SQLite database found at {path} — nothing to migrate.")
        sys.exit(1)
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    return conn


def _migrate_table(sqlite_conn, pg_conn, table: str, columns: list[str]) -> int:
    rows = sqlite_conn.execute(f"SELECT {', '.join(columns)} FROM {table}").fetchall()
    if not rows:
        print(f"  {table}: 0 rows (nothing to copy)")
        return 0

    placeholders = ", ".join(["%s"] * len(columns))
    col_list = ", ".join(columns)
    # Assumes first column is a usable conflict target (id / primary key) for
    # every table this script touches — true for users/api_keys/audit_log
    # (id), and usage_daily uses its composite key below as a special case.
    insert_sql = (
        f"INSERT INTO {table} ({col_list}) VALUES ({placeholders}) "
        f"ON CONFLICT DO NOTHING"
    )

    copied = 0
    with pg_conn.cursor() as cur:
        for row in rows:
            cur.execute(insert_sql, tuple(row[c] for c in columns))
            copied += cur.rowcount
    pg_conn.commit()
    print(f"  {table}: {copied}/{len(rows)} rows copied (rest already present)")
    return copied


def migrate() -> None:
    dsn = os.environ.get("ORCA_DATABASE_URL")
    if not dsn:
        print("ORCA_DATABASE_URL not set — nothing to migrate to. Set it and re-run.")
        sys.exit(1)

    import psycopg

    # Import orca.auth.db first — its module-level init_db() creates the
    # target schema (users/usage_daily/audit_log tables) if it doesn't exist
    # yet. Without this, the INSERTs below hit a fresh Postgres DB with no
    # tables at all.
    import orca.auth.db  # noqa: F401 — import for its init_db() side effect

    sqlite_conn = _sqlite_source_conn()
    pg_conn = psycopg.connect(dsn)

    print("Migrating SQLite -> Postgres")
    print(f"  source: {ORCA_HOME / 'auth.db'}")
    print(f"  target: {dsn.split('@')[-1] if '@' in dsn else dsn}\n")

    _migrate_table(sqlite_conn, pg_conn, "users",
                   ["id", "email", "name", "password_hash", "tier", "role", "created_at", "verified"])

    _migrate_table(sqlite_conn, pg_conn, "usage_daily",
                   ["user_id", "date", "messages", "ultra_runs"])

    # audit_log: copy hash-chain fields byte-for-byte, never recompute
    _migrate_table(sqlite_conn, pg_conn, "audit_log",
                   ["id", "seq", "user_id", "event", "detail", "ip", "created_at",
                    "prev_hash", "entry_hash", "signature"])

    try:
        _migrate_table(sqlite_conn, pg_conn, "api_keys",
                       ["id", "user_id", "key_hash", "name", "created_at", "last_used", "revoked"])
    except Exception as e:
        print(f"  api_keys: skipped ({e}) — table may not exist in source DB yet")

    pg_conn.close()
    sqlite_conn.close()

    print("\nVerifying migrated audit chain integrity...")
    # Re-import audit AFTER migration so it picks up ORCA_DATABASE_URL and
    # reads from Postgres, not the SQLite file we just migrated from.
    import orca.audit as audit
    report = audit.verify_chain()
    if report["valid"]:
        print(f"  ✓ Chain valid — {report['entries_checked']} entries verified, no breaks.")
    else:
        print(f"  ✗ Chain INVALID — {len(report['breaks'])} break(s) found:")
        for b in report["breaks"]:
            print(f"    seq {b['seq']}: {b['reason']} — {b['detail']}")
        print("\n  Do not delete the source SQLite database until this is resolved.")
        sys.exit(1)

    print("\nMigration complete. Source SQLite file left untouched at:")
    print(f"  {ORCA_HOME / 'auth.db'}")
    print("Delete it manually once you've confirmed the production deployment is stable.")


if __name__ == "__main__":
    migrate()
