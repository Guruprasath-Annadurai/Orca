"""
Automated backup / disaster recovery — real gap this closes: SQLite/Postgres
had ZERO backup automation. One bad `rm`, disk failure, or corrupted write
loses the auth DB (users, audit log, everything) with no way back.

HONEST SCOPE:
  - This is a backup/prune tool, meant to be invoked by an external
    scheduler (cron, systemd timer, a scheduled CI job) — it does NOT
    schedule itself inside the running server process. That's deliberate:
    an in-process scheduler dies when the process dies/restarts/crashes,
    which is exactly when you most need backups to keep running. Standard
    ops practice is external scheduling; this tool is the thing THAT gets
    scheduled, not a scheduler itself.
  - SQLite backup uses sqlite3's own online backup API (Connection.backup()),
    not a raw file copy — a raw `cp` of a live SQLite file (especially in
    WAL mode, which orca/auth/db.py uses) can capture a half-written,
    inconsistent state. The backup API produces a consistent snapshot even
    while the source DB is being actively written to.
  - Postgres backup shells out to `pg_dump` — requires it installed on the
    machine running the backup (checked explicitly, raises a clear error
    if missing rather than silently producing an empty/broken dump).
  - Restore is documented and provided for SQLite (the common single-user
    case — copy the backup file back over the live DB). Postgres restore
    uses standard `pg_restore`/`psql < dump.sql` — documented, not
    reimplemented, since re-inventing Postgres's own restore tooling would
    be needless risk for something Postgres already does correctly.
"""
from __future__ import annotations

import shutil
import sqlite3
import subprocess
import time
from pathlib import Path

from orca.config import ORCA_HOME
from orca.auth.db import AUTH_DB, BACKEND

BACKUP_DIR = ORCA_HOME / "backups"
BACKUP_DIR.mkdir(parents=True, exist_ok=True)


def backup_sqlite() -> Path:
    """
    Consistent snapshot via SQLite's own backup API — safe to run while the
    live DB is being written to (this project uses WAL mode, see
    orca/auth/db.py), unlike a raw file copy which could capture a
    half-written state.
    """
    if not AUTH_DB.exists():
        raise FileNotFoundError(f"No SQLite database found at {AUTH_DB} — nothing to back up.")

    ts = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    dest = BACKUP_DIR / f"auth_backup_{ts}.db"

    source_conn = sqlite3.connect(str(AUTH_DB))
    dest_conn = sqlite3.connect(str(dest))
    try:
        source_conn.backup(dest_conn)
    finally:
        source_conn.close()
        dest_conn.close()

    return dest


def backup_postgres(database_url: str) -> Path:
    """Shells out to pg_dump — requires it installed. Custom format (-Fc), restorable via pg_restore."""
    if shutil.which("pg_dump") is None:
        raise RuntimeError(
            "pg_dump not found on PATH. Install PostgreSQL client tools "
            "(e.g. `brew install postgresql` or your distro's postgresql-client "
            "package) before running Postgres backups."
        )

    ts = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    dest = BACKUP_DIR / f"auth_backup_{ts}.pgdump"

    result = subprocess.run(
        ["pg_dump", database_url, "-Fc", "-f", str(dest)],
        capture_output=True, text=True, timeout=300,
    )
    if result.returncode != 0:
        raise RuntimeError(f"pg_dump failed: {result.stderr}")

    return dest


def run_backup() -> dict:
    """Dispatches to the right backup method based on the active DB backend."""
    import os

    if BACKEND == "postgres":
        database_url = os.environ.get("ORCA_DATABASE_URL", "")
        path = backup_postgres(database_url)
        backend = "postgres"
    else:
        path = backup_sqlite()
        backend = "sqlite"

    size_bytes = path.stat().st_size
    return {
        "backend": backend,
        "path": str(path),
        "size_bytes": size_bytes,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
    }


def list_backups() -> list[dict]:
    backups = []
    for path in sorted(BACKUP_DIR.glob("auth_backup_*"), key=lambda p: p.stat().st_mtime, reverse=True):
        backups.append({
            "path": str(path),
            "size_bytes": path.stat().st_size,
            "modified_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(path.stat().st_mtime)),
        })
    return backups


def prune_old_backups(keep_last_n: int = 7) -> dict:
    """Deletes backups beyond the retention count, oldest first. Never deletes the backup just created this run."""
    backups = list_backups()  # already sorted newest-first
    to_delete = backups[keep_last_n:]

    deleted = []
    for b in to_delete:
        try:
            Path(b["path"]).unlink()
            deleted.append(b["path"])
        except Exception:
            pass  # a failed delete of an old backup isn't worth failing the whole prune run over

    return {"kept": min(len(backups), keep_last_n), "deleted": deleted, "deleted_count": len(deleted)}


def restore_sqlite(backup_path: str, confirm: bool = False) -> dict:
    """
    Restores a SQLite backup over the LIVE database — destructive, requires
    explicit confirm=True. Does not restore Postgres (see module docstring
    for why — use pg_restore directly for that backend).
    """
    if BACKEND != "sqlite":
        raise RuntimeError(f"Current backend is '{BACKEND}', not sqlite. This function only restores SQLite backups.")
    if not confirm:
        raise ValueError("Restoring will OVERWRITE the live database. Call with confirm=True to proceed.")

    src = Path(backup_path)
    if not src.exists():
        raise FileNotFoundError(f"Backup file not found: {backup_path}")

    # Back up the CURRENT (about-to-be-overwritten) state first, so a bad
    # restore choice is itself recoverable rather than a second data-loss event.
    pre_restore_backup = backup_sqlite()

    shutil.copy2(src, AUTH_DB)

    return {
        "restored_from": str(src),
        "pre_restore_backup": str(pre_restore_backup),
        "restored_at": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
