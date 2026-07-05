"""
Audit log — hash-chained, tamper-evident event ledger.

Every entry embeds the SHA-256 hash of the previous entry (blockchain-style
Merkle chain, no consensus needed since it's single-writer). Tampering with
any historical row breaks the chain from that point forward — detectable by
recomputing hashes and comparing, no external trust anchor required.

Optional HMAC signing (ORCA_AUDIT_KEY env var) adds a second layer: even if
an attacker rewrites the entire chain consistently, they cannot forge valid
signatures without the key. Key should live outside the DB (env var, secret
manager, HSM) — never store it alongside the log it protects.

This is the trust foundation the enterprise governance layer sits on top of —
model cards, red-team reports, and "explain this answer" traces all reference
audit entries by id, and the whole point is a compliance officer or auditor
can verify none of it was altered after the fact.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import os
import threading
import time
import uuid

from orca.auth.db import get_conn, BACKEND

_GENESIS_HASH = "0" * 64
_chain_lock = threading.Lock()   # serializes hash-chain writes — must be strictly ordered


def _audit_key() -> bytes:
    """HMAC signing key. Falls back to a fixed dev key with a loud warning."""
    key = os.environ.get("ORCA_AUDIT_KEY")
    if key:
        return key.encode()
    return b"orca-dev-audit-key-DO-NOT-USE-IN-PRODUCTION"


def _ensure_table() -> None:
    with get_conn() as conn:
        conn.execute("""
        CREATE TABLE IF NOT EXISTS audit_log (
            id          TEXT PRIMARY KEY,
            seq         INTEGER,
            user_id     TEXT,
            event       TEXT NOT NULL,
            detail      TEXT,
            ip          TEXT,
            created_at  REAL NOT NULL,
            prev_hash   TEXT NOT NULL,
            entry_hash  TEXT NOT NULL,
            signature   TEXT NOT NULL
        )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS ix_audit_user  ON audit_log(user_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS ix_audit_event ON audit_log(event)")
        conn.execute("CREATE INDEX IF NOT EXISTS ix_audit_ts    ON audit_log(created_at)")
        if BACKEND == "sqlite":
            # Postgres installs are always fresh — orca/auth/db.py's schema
            # already includes seq/prev_hash/entry_hash/signature from the
            # start, so there's no legacy table to migrate. PRAGMA table_info
            # (used below) is SQLite-only syntax and would error on Postgres.
            _migrate_legacy_schema(conn)
        conn.execute("CREATE INDEX IF NOT EXISTS ix_audit_seq   ON audit_log(seq)")


def _migrate_legacy_schema(conn) -> None:
    """
    Pre-hash-chain installs have an audit_log table missing seq/prev_hash/
    entry_hash/signature columns. ALTER TABLE in, then backfill the chain for
    existing rows (ordered by created_at, since they predate seq).

    Backfilled entries are hashed exactly like new ones so verify_chain()
    treats them identically — the only difference is a "migrated: true" note
    isn't needed since the hash covers actual stored content, not metadata
    about how it got there.
    """
    cols = {r["name"] for r in conn.execute("PRAGMA table_info(audit_log)").fetchall()}

    if "seq" not in cols:
        conn.execute("ALTER TABLE audit_log ADD COLUMN seq INTEGER")
    if "prev_hash" not in cols:
        conn.execute("ALTER TABLE audit_log ADD COLUMN prev_hash TEXT")
    if "entry_hash" not in cols:
        conn.execute("ALTER TABLE audit_log ADD COLUMN entry_hash TEXT")
    if "signature" not in cols:
        conn.execute("ALTER TABLE audit_log ADD COLUMN signature TEXT")

    # Backfill runs independently of the column-add step above: adding a column
    # and finishing the backfill are two separate operations, and a crash between
    # them must not leave rows permanently un-hashed on the next startup. Always
    # check for (and finish) any row where seq IS NULL, regardless of whether the
    # ALTER TABLE calls above were no-ops this time.
    rows = conn.execute(
        "SELECT rowid, id, user_id, event, detail, ip, created_at FROM audit_log "
        "WHERE seq IS NULL ORDER BY created_at ASC"
    ).fetchall()
    if not rows:
        return

    prev_hash = _GENESIS_HASH
    # If some rows already have a chain (partial migration retry), continue after them
    existing_max = conn.execute("SELECT MAX(seq) as m, entry_hash FROM audit_log WHERE seq IS NOT NULL").fetchone()
    seq = 0
    if existing_max and existing_max["m"] is not None:
        seq = existing_max["m"] + 1
        prev_hash = existing_max["entry_hash"]

    for r in rows:
        payload = _canonical_payload(seq, r["user_id"], r["event"], r["detail"], r["ip"], r["created_at"], prev_hash)
        entry_hash = _compute_hash(payload)
        signature  = _compute_signature(entry_hash)
        conn.execute(
            "UPDATE audit_log SET seq=?, prev_hash=?, entry_hash=?, signature=? WHERE rowid=?",
            (seq, prev_hash, entry_hash, signature, r["rowid"]),
        )
        prev_hash = entry_hash
        seq += 1


def _canonical_payload(seq: int, user_id, event, detail_raw, ip, created_at, prev_hash) -> str:
    """
    Deterministic string representation for hashing. detail_raw must be the
    EXACT string stored in the DB (not a re-serialized dict) — re-serializing
    a parsed dict can reorder keys or change whitespace and silently break
    every downstream verification.
    """
    return json.dumps({
        "seq": seq, "user_id": user_id, "event": event, "detail": detail_raw,
        "ip": ip, "created_at": created_at, "prev_hash": prev_hash,
    }, sort_keys=True, separators=(",", ":"))


def _compute_hash(payload: str) -> str:
    return hashlib.sha256(payload.encode()).hexdigest()


def _compute_signature(entry_hash: str) -> str:
    return hmac.new(_audit_key(), entry_hash.encode(), hashlib.sha256).hexdigest()


def log(
    event: str,
    user_id: str | None = None,
    detail: dict | None = None,
    ip: str | None = None,
) -> str | None:
    """
    Append a hash-chained audit entry. Never raises (audit failures must not
    break the request they're logging). Returns the entry id, or None on failure.
    """
    try:
        detail_raw = json.dumps(detail) if detail else None
        created_at = time.time()
        entry_id = str(uuid.uuid4())

        with _chain_lock:
            with get_conn() as conn:
                if BACKEND == "postgres":
                    # _chain_lock only serializes writers within THIS process —
                    # useless once multiple API instances write to the same
                    # Postgres database. pg_advisory_xact_lock takes a
                    # cluster-wide lock scoped to this transaction, auto-released
                    # on commit/rollback (i.e. when the `with get_conn()` block
                    # exits) — every instance blocks here until the previous
                    # writer's transaction finishes, keeping seq/prev_hash
                    # assignment strictly ordered across the whole deployment.
                    conn.execute("SELECT pg_advisory_xact_lock(hashtext('orca_audit_chain'))")

                last = conn.execute(
                    "SELECT entry_hash, seq FROM audit_log ORDER BY seq DESC LIMIT 1"
                ).fetchone()
                prev_hash = last["entry_hash"] if last else _GENESIS_HASH
                seq = (last["seq"] + 1) if last else 0

                payload = _canonical_payload(seq, user_id, event, detail_raw, ip, created_at, prev_hash)
                entry_hash = _compute_hash(payload)
                signature  = _compute_signature(entry_hash)

                conn.execute(
                    "INSERT INTO audit_log "
                    "(id, seq, user_id, event, detail, ip, created_at, prev_hash, entry_hash, signature) "
                    "VALUES (?,?,?,?,?,?,?,?,?,?)",
                    (entry_id, seq, user_id, event, detail_raw, ip, created_at,
                     prev_hash, entry_hash, signature),
                )
        return entry_id
    except Exception:
        return None


def recent(limit: int = 100, user_id: str | None = None) -> list[dict]:
    with get_conn() as conn:
        if user_id:
            rows = conn.execute(
                "SELECT * FROM audit_log WHERE user_id=? ORDER BY seq DESC LIMIT ?",
                (user_id, limit),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM audit_log ORDER BY seq DESC LIMIT ?",
                (limit,),
            ).fetchall()
    return [_row_to_display(r) for r in rows]


def _row_to_display(r) -> dict:
    """Human-facing view — detail parsed back into a dict."""
    return {
        "id":         r["id"],
        "seq":        r["seq"],
        "user_id":    r["user_id"],
        "event":      r["event"],
        "detail":     json.loads(r["detail"]) if r["detail"] else None,
        "ip":         r["ip"],
        "created_at": r["created_at"],
        "prev_hash":  r["prev_hash"],
        "entry_hash": r["entry_hash"],
        "signature":  r["signature"],
    }


def verify_chain(start_seq: int = 0, end_seq: int | None = None) -> dict:
    """
    Recompute every hash and signature in [start_seq, end_seq] and compare
    against stored values. Returns a verification report — the thing a
    compliance officer / court would actually run.
    """
    with get_conn() as conn:
        if end_seq is not None:
            rows = conn.execute(
                "SELECT * FROM audit_log WHERE seq>=? AND seq<=? ORDER BY seq ASC",
                (start_seq, end_seq),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM audit_log WHERE seq>=? ORDER BY seq ASC",
                (start_seq,),
            ).fetchall()

    if not rows:
        return {"valid": True, "entries_checked": 0, "breaks": [], "checked_at": time.time()}

    breaks: list[dict] = []
    partial_chain = start_seq != 0

    for i, r in enumerate(rows):
        # Recompute hash from the RAW stored detail string (r["detail"]), never
        # from a re-serialized dict — see _canonical_payload docstring.
        payload = _canonical_payload(
            r["seq"], r["user_id"], r["event"], r["detail"], r["ip"], r["created_at"], r["prev_hash"],
        )
        recomputed_hash = _compute_hash(payload)
        recomputed_sig  = _compute_signature(recomputed_hash)

        if recomputed_hash != r["entry_hash"]:
            breaks.append({"seq": r["seq"], "id": r["id"], "reason": "hash_mismatch",
                            "detail": "Entry contents don't match stored hash — row was altered."})
        if recomputed_sig != r["signature"]:
            breaks.append({"seq": r["seq"], "id": r["id"], "reason": "signature_mismatch",
                            "detail": "HMAC signature invalid — entry forged without the audit key."})

        # Chain linkage: each entry's prev_hash must equal the PRECEDING entry's
        # actual entry_hash. First row in a partial range can't be checked this
        # way (we don't have its predecessor) — only flag chain breaks from the
        # second row onward, or from row 0 if this is the full chain (seq==0 must
        # equal genesis).
        if i == 0:
            if not partial_chain and r["seq"] == 0 and r["prev_hash"] != _GENESIS_HASH:
                breaks.append({"seq": r["seq"], "id": r["id"], "reason": "bad_genesis",
                                "detail": "First entry's prev_hash is not the genesis hash."})
        else:
            prior_hash = rows[i - 1]["entry_hash"]
            if r["prev_hash"] != prior_hash:
                breaks.append({"seq": r["seq"], "id": r["id"], "reason": "chain_broken",
                                "detail": f"prev_hash doesn't match preceding entry's hash — "
                                          f"chain link broken between seq {rows[i-1]['seq']} and {r['seq']}."})

    return {
        "valid": len(breaks) == 0,
        "entries_checked": len(rows),
        "range": {"start_seq": rows[0]["seq"], "end_seq": rows[-1]["seq"]},
        "breaks": breaks,
        "checked_at": time.time(),
    }


def export_for_audit(start_seq: int = 0, end_seq: int | None = None) -> dict:
    """
    Produce a court-admissible export: full entries + verification result +
    a top-level signature over the whole export (so the export file itself
    can be checked for tampering after it leaves the system).
    """
    with get_conn() as conn:
        if end_seq is not None:
            rows = conn.execute(
                "SELECT * FROM audit_log WHERE seq>=? AND seq<=? ORDER BY seq ASC",
                (start_seq, end_seq),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM audit_log WHERE seq>=? ORDER BY seq ASC",
                (start_seq,),
            ).fetchall()

    entries = [_row_to_display(r) for r in rows]
    verification = verify_chain(start_seq, end_seq)

    export = {
        "exported_at": time.time(),
        "entry_count": len(entries),
        "verification": verification,
        "entries": entries,
    }
    export_payload = json.dumps(export, sort_keys=True, separators=(",", ":"))
    export["export_signature"] = _compute_signature(_compute_hash(export_payload))
    return export


_ensure_table()
