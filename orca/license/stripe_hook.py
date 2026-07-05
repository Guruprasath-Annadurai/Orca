"""
Stripe webhook handler for automatic license fulfillment.

Flow:
  Customer pays on Stripe →
  Stripe calls POST /webhook/stripe →
  We verify HMAC signature →
  checkout.session.completed event →
  Generate license key →
  Email customer →
  Log to ~/.orca/logs/license_log.jsonl

Required env vars (.env):
  STRIPE_WEBHOOK_SECRET   — from Stripe dashboard (whsec_...)
  ORCA_LICENSE_SECRET     — used to sign keys
  STRIPE_PRICE_PRO        — Price ID for Pro monthly  (price_...)
  STRIPE_PRICE_PRO_YEAR   — Price ID for Pro annual
  STRIPE_PRICE_ENT        — Price ID for Enterprise monthly
  STRIPE_PRICE_ENT_YEAR   — Price ID for Enterprise annual
  SMTP_HOST / SMTP_USER / SMTP_PASS — for email delivery

Stripe metadata overrides (set in Payment Link / Product):
  tier  — "pro" | "enterprise"
  seats — number of seats (integer)
  days  — license duration in days (0 = lifetime)
"""
from __future__ import annotations

import hashlib
import hmac as _hmac
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path


# ─── Price ID → license params mapping ────────────────────────────────────────

def _price_to_params(price_id: str) -> tuple[str, int, int]:
    """Returns (tier, seats, days) from a Stripe Price ID."""
    pro_m  = os.environ.get("STRIPE_PRICE_PRO", "")
    pro_y  = os.environ.get("STRIPE_PRICE_PRO_YEAR", "")
    ent_m  = os.environ.get("STRIPE_PRICE_ENT", "")
    ent_y  = os.environ.get("STRIPE_PRICE_ENT_YEAR", "")

    if price_id == pro_m:   return "pro",        1,  31
    if price_id == pro_y:   return "pro",        1, 365
    if price_id == ent_m:   return "enterprise", 5,  31
    if price_id == ent_y:   return "enterprise", 5, 365

    # Unknown price — default to Pro monthly (safe fallback)
    return "pro", 1, 31


# ─── Main event handler ────────────────────────────────────────────────────────

def handle_stripe_event(payload: bytes, sig_header: str) -> dict:
    """
    Process a raw Stripe webhook request.

    Returns:
      {"status": "ok",      "key": ..., "email": ..., ...}
      {"status": "ignored", "type": event_type}
      raises ValueError on invalid signature
    """
    secret = os.environ.get("STRIPE_WEBHOOK_SECRET", "")
    event  = _verify_signature(payload, sig_header, secret)

    if event["type"] == "checkout.session.completed":
        return _handle_checkout_completed(event)
    if event["type"] in ("customer.subscription.deleted", "invoice.payment_failed"):
        return _handle_subscription_downgrade(event)

    return {"status": "ignored", "type": event["type"]}


def _handle_checkout_completed(event: dict) -> dict:
    session  = event["data"]["object"]
    metadata = session.get("metadata") or {}

    email = (
        session.get("customer_details", {}).get("email")
        or session.get("customer_email")
        or metadata.get("email", "")
    )

    # Allow metadata overrides (set in Stripe Payment Link or Product metadata)
    tier_override  = metadata.get("tier", "")
    seats_override = int(metadata.get("seats", 0))
    days_override  = int(metadata.get("days", 0))
    price_id       = metadata.get("price_id", "")

    if tier_override:
        tier  = tier_override
        seats = seats_override or 1
        days  = days_override  or 365
    else:
        tier, seats, days = _price_to_params(price_id)

    # ── Upgrade the web account, if this checkout came from a logged-in user ──
    # client_reference_id is set by POST /api/billing/checkout (orca/serve/api.py)
    # to the authenticated user's id. Payment Links created directly in the
    # Stripe dashboard (no app-created Checkout Session) won't have this —
    # those customers only get the offline license-key flow below, same as
    # before this fix. Each half is wrapped independently so a failure in one
    # (e.g. DB unavailable) doesn't block the other from completing.
    user_id = session.get("client_reference_id") or metadata.get("user_id", "")
    account_updated = False
    if user_id:
        try:
            from orca.auth.store import set_user_tier, set_stripe_customer_id
            set_user_tier(user_id, tier)
            customer_id = session.get("customer")
            if customer_id:
                set_stripe_customer_id(user_id, customer_id)
            account_updated = True
        except Exception:
            pass  # web account update failed — license key issuance still proceeds below

    # Generate the offline license key (desktop/CLI activation flow — unchanged)
    from orca.license.keys import generate_key
    key = generate_key(tier=tier, seats=seats, days=days)

    # Persist to admin log
    _append_log(key, tier, seats, days, email, session.get("id", ""))

    # Email the customer
    email_sent = False
    if email:
        from orca.license.mailer import send_license_email
        email_sent = send_license_email(email, key, tier, seats, days)

    return {
        "status":          "ok",
        "key":             key,
        "tier":            tier,
        "seats":           seats,
        "days":            days,
        "email":           email,
        "email_sent":      email_sent,
        "account_updated": account_updated,
    }


def _handle_subscription_downgrade(event: dict) -> dict:
    """
    customer.subscription.deleted (cancellation) or invoice.payment_failed
    (card declined) — downgrade the web account back to free. Without this,
    a cancelled or failed subscription left the account at its paid tier
    forever, since nothing ever downgraded it.
    """
    obj         = event["data"]["object"]
    customer_id = obj.get("customer", "")
    if not customer_id:
        return {"status": "ignored", "type": event["type"], "reason": "no customer id on event"}

    try:
        from orca.auth.store import get_user_by_stripe_customer_id, set_user_tier
        user = get_user_by_stripe_customer_id(customer_id)
        if not user:
            return {"status": "ignored", "type": event["type"], "reason": "no matching account"}
        set_user_tier(user.id, "free")
        return {"status": "ok", "type": event["type"], "user_id": user.id, "downgraded_to": "free"}
    except Exception as e:
        return {"status": "error", "type": event["type"], "reason": str(e)}


# ─── Signature verification ────────────────────────────────────────────────────

def _verify_signature(payload: bytes, sig_header: str, secret: str) -> dict:
    """
    Verify Stripe's webhook signature.
    Raises ValueError on failure.
    """
    if not secret:
        # Dev mode — skip verification
        return json.loads(payload)

    parts: dict[str, str] = {}
    for chunk in sig_header.split(","):
        k, _, v = chunk.partition("=")
        parts[k.strip()] = v.strip()

    timestamp = parts.get("t", "0")
    v1_sig    = parts.get("v1", "")

    if abs(time.time() - int(timestamp)) > 300:
        raise ValueError("Webhook timestamp too old — possible replay attack")

    signed = timestamp.encode() + b"." + payload
    expected = _hmac.new(secret.encode(), signed, hashlib.sha256).hexdigest()

    if not _hmac.compare_digest(expected, v1_sig):
        raise ValueError("Stripe signature verification failed")

    return json.loads(payload)


# ─── Admin log ────────────────────────────────────────────────────────────────

def _append_log(
    key: str, tier: str, seats: int, days: int,
    email: str, stripe_session: str,
) -> None:
    try:
        from orca.config import ORCA_HOME
        log_path = ORCA_HOME / "logs" / "license_log.jsonl"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "ts":             datetime.now(tz=timezone.utc).isoformat(),
            "key":            key,
            "tier":           tier,
            "seats":          seats,
            "days":           days,
            "email":          email,
            "stripe_session": stripe_session,
        }
        with open(log_path, "a") as f:
            f.write(json.dumps(record) + "\n")
    except Exception:
        pass  # log failure is non-fatal


# ─── Admin helpers ─────────────────────────────────────────────────────────────

def list_issued_keys(n: int = 50) -> list[dict]:
    """Read the last n issued license records from the admin log."""
    try:
        from orca.config import ORCA_HOME
        log_path = ORCA_HOME / "logs" / "license_log.jsonl"
        if not log_path.exists():
            return []
        lines = log_path.read_text().strip().splitlines()
        records = [json.loads(ln) for ln in lines if ln.strip()]
        return records[-n:]
    except Exception:
        return []
