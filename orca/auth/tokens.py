"""Short-lived HMAC-signed tokens for email verification and password reset.

Tokens are URL-safe strings: base64(json_payload).hmac_hex
No DB storage needed — expiry is embedded in the payload.
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import time
from typing import Optional

_SECRET = os.environ.get("ORCA_AUTH_SECRET", "dev-secret-change-me")


def _encode(payload: dict) -> str:
    data = base64.urlsafe_b64encode(json.dumps(payload, separators=(",", ":")).encode()).decode().rstrip("=")
    sig  = hmac.new(_SECRET.encode(), data.encode(), hashlib.sha256).hexdigest()
    return f"{data}.{sig}"


def _decode(token: str) -> Optional[dict]:
    try:
        data, sig = token.rsplit(".", 1)
        expected  = hmac.new(_SECRET.encode(), data.encode(), hashlib.sha256).hexdigest()
        if not hmac.compare_digest(sig, expected):
            return None
        padding = 4 - len(data) % 4
        payload = json.loads(base64.urlsafe_b64decode(data + "=" * padding))
        if payload.get("exp", 0) < time.time():
            return None
        return payload
    except Exception:
        return None


def make_verification_token(user_id: str, email: str) -> str:
    return _encode({"sub": user_id, "email": email, "type": "verify", "exp": int(time.time()) + 86400})


def make_reset_token(user_id: str, email: str) -> str:
    return _encode({"sub": user_id, "email": email, "type": "reset",  "exp": int(time.time()) + 3600})


def make_2fa_pending_token(user_id: str, email: str) -> str:
    """
    Issued after password check passes but BEFORE the TOTP code is verified —
    proves "this request already knows the password" without granting a real
    session token. Short expiry (5 min) since it's meant to be exchanged
    immediately for the real token via /api/auth/2fa/verify-login.
    """
    return _encode({"sub": user_id, "email": email, "type": "2fa_pending", "exp": int(time.time()) + 300})


def verify_token(token: str, expected_type: str) -> Optional[dict]:
    """Returns payload dict if valid and type matches, else None."""
    payload = _decode(token)
    if payload and payload.get("type") == expected_type:
        return payload
    return None
