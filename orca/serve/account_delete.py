"""
Right-to-delete — orchestrates a full account deletion across every store
that holds data tied to a user, not just the users table row.

Real architectural gap this closes: chat history, uploaded documents, and
memory files were keyed purely by session_id, with no persisted link back
to an owning user_id. orca/auth/store.py's user_sessions table (populated
by _get_session() whenever an authenticated request creates/touches a
session) is what makes "find everything this account touched" possible at
all — without it, this module could only delete the account row itself and
silently leave chat history/documents behind.

What this deletes:
  - The user's account row, API keys, usage records (orca.auth.store)
  - Every session this account is known to have touched:
      - Episodic memory file (orca/brain/memory.py's EpisodicMemory)
      - DocStore vectors + doc registry entries (orca/docs/store.py)
      - Knowledge graph entities/relationships (orca/brain/knowledge_graph.py)
      - Session title entry (orca/serve/api.py's session_titles.json)
      - Redis session state, if enabled (orca/serve/session_store.py)
  - Attempts to cancel any active Stripe subscription tied to this account
    (best-effort — logged, not fatal if it fails)

What this does NOT delete, stated plainly (same tension already documented
in legal/PRIVACY_POLICY.md §5):
  - Historical audit log entries referencing this user_id. The audit log's
    entire value is being a tamper-evident record of what happened — a
    log you can retroactively edit isn't tamper-evident. Deleting entries
    would break the hash chain for every entry after it. The user_id in
    old audit entries becomes a dangling reference to a deleted account,
    not a live PII record — a policy/legal judgment call for the operator's
    jurisdiction, not something resolved automatically here.
  - Sessions the user touched WHILE UNAUTHENTICATED (anonymous chat before
    ever logging in) — those were never recorded in user_sessions because
    there was no user_id to record them against at the time. This is a
    real, current limitation, not swept under the rug.
"""
from __future__ import annotations

import os
import json
from pathlib import Path

from orca.auth.store import get_user_session_ids, delete_user_account_records, get_stripe_customer_id
from orca.docs.store import unregister_doc, list_docs, DocStore
from orca.brain.memory import EpisodicMemory
from orca.brain.knowledge_graph import KnowledgeGraph
from orca.serve import session_store
from orca.config import ORCA_HOME, CONFIG


def _delete_session_data(session_id: str) -> dict:
    """Best-effort cleanup of everything scoped to one session_id. Never raises — one failed store shouldn't block cleanup of the rest."""
    result = {"session_id": session_id, "memory_deleted": False, "docs_deleted": 0, "redis_deleted": False, "knowledge_graph_deleted": False}

    try:
        ep = EpisodicMemory(session_id)
        if ep.path.exists():
            ep.path.unlink()
            result["memory_deleted"] = True
    except Exception as e:
        result["memory_error"] = str(e)

    try:
        docs = list_docs(session_id)
        if docs:
            store = DocStore(session_id=session_id, ollama_host=CONFIG.ollama.host)
            for d in docs:
                store.delete_doc(d["doc_id"])
                unregister_doc(session_id, d["doc_id"])
            result["docs_deleted"] = len(docs)
    except Exception as e:
        result["docs_error"] = str(e)

    try:
        kg = KnowledgeGraph(session_id)
        had_data = kg.count()["entities"] > 0
        kg.clear()
        result["knowledge_graph_deleted"] = had_data
    except Exception as e:
        result["knowledge_graph_error"] = str(e)

    try:
        session_store.delete_session(session_id)
        result["redis_deleted"] = session_store.enabled()
    except Exception as e:
        result["redis_error"] = str(e)

    return result


def _remove_session_title(session_id: str) -> None:
    """session_titles.json lives in orca/serve/api.py's module state, not a
    shared store — read/write the file directly rather than importing api.py
    (which would create a circular import: api.py imports this module)."""
    titles_path = ORCA_HOME / "session_titles.json"
    try:
        if not titles_path.exists():
            return
        titles = json.loads(titles_path.read_text())
        if session_id in titles:
            del titles[session_id]
            titles_path.write_text(json.dumps(titles))
    except Exception:
        pass


def _cancel_stripe_subscriptions(customer_id: str) -> dict:
    """Best-effort — if Stripe isn't configured or the call fails, deletion still proceeds. A failed cancellation attempt is not a reason to block account deletion."""
    stripe_secret = os.environ.get("STRIPE_SECRET_KEY")
    if not stripe_secret:
        return {"attempted": False, "reason": "STRIPE_SECRET_KEY not configured"}

    try:
        import stripe
        stripe.api_key = stripe_secret
        subs = stripe.Subscription.list(customer=customer_id, status="active", limit=10)
        cancelled = []
        for sub in subs.data:
            stripe.Subscription.cancel(sub.id)
            cancelled.append(sub.id)
        return {"attempted": True, "cancelled_subscription_ids": cancelled}
    except Exception as e:
        return {"attempted": True, "error": str(e)}


def delete_account(user_id: str) -> dict:
    """
    Full account deletion. Returns a report of what was actually deleted —
    not just "success: true", since a partial failure in one store
    shouldn't be hidden behind an overall success flag.
    """
    session_ids = get_user_session_ids(user_id)
    session_results = []
    for sid in session_ids:
        session_results.append(_delete_session_data(sid))
        _remove_session_title(sid)

    stripe_result = None
    customer_id = get_stripe_customer_id(user_id)
    if customer_id:
        stripe_result = _cancel_stripe_subscriptions(customer_id)

    # Account row + api_keys + usage_daily + user_sessions — last, so if
    # anything above fails we still know which sessions belonged to this
    # user (the user_sessions rows aren't gone yet until this call).
    delete_user_account_records(user_id)

    return {
        "user_id": user_id,
        "sessions_processed": len(session_ids),
        "session_results": session_results,
        "stripe_cancellation": stripe_result,
        "note": (
            "Historical audit log entries referencing this account are NOT deleted "
            "(tamper-evident by design — see legal/PRIVACY_POLICY.md §5). Sessions used "
            "before ever logging in (anonymous/unauthenticated) are not tracked and "
            "cannot be cascaded to by this deletion."
        ),
    }
