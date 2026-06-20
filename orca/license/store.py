"""
Local license store — persists activated license to ~/.orca/license.json.

Decoupled from key validation so the store can be read without the secret.
Re-validates on every load to catch expiry changes.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from orca.license.keys import LicenseKey, validate_key

LICENSE_PATH = Path.home() / ".orca" / "license.json"


# ─── Read / write ──────────────────────────────────────────────────────────────

def load_record() -> Optional[dict]:
    """Return the raw stored record, or None if not activated."""
    if not LICENSE_PATH.exists():
        return None
    try:
        return json.loads(LICENSE_PATH.read_text())
    except Exception:
        return None


def save_license(lk: LicenseKey, email: str = "") -> None:
    """Persist a validated license."""
    LICENSE_PATH.parent.mkdir(parents=True, exist_ok=True)
    LICENSE_PATH.write_text(json.dumps({
        "key":          lk.raw,
        "tier":         lk.tier,
        "seats":        lk.seats,
        "expiry":       lk.expiry.isoformat() if lk.expiry else None,
        "activated_at": datetime.now(tz=timezone.utc).isoformat(),
        "email":        email,
    }, indent=2))


def clear_license() -> None:
    """Remove the stored license."""
    if LICENSE_PATH.exists():
        LICENSE_PATH.unlink()


# ─── Active license helpers ────────────────────────────────────────────────────

def get_active_license() -> Optional[LicenseKey]:
    """
    Return the active license if stored and currently valid.
    Re-validates the key on every call (catches expiry without network).
    Returns None if not activated or expired.
    """
    record = load_record()
    if not record:
        return None
    lk = validate_key(record["key"])
    return lk if lk.valid else None


def current_tier() -> str:
    """Returns the highest active tier: 'enterprise', 'pro', or 'free'."""
    lk = get_active_license()
    return lk.tier if lk else "free"


def activation_email() -> str:
    record = load_record()
    return record.get("email", "") if record else ""
