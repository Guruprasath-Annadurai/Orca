"""Password hashing + stateless token signing — zero external deps."""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import secrets
import time


# ── Password hashing (PBKDF2-SHA256) ─────────────────────────────────────────

def hash_password(password: str) -> str:
    salt = secrets.token_hex(16)
    dk = hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(), 260_000)
    return f"pbkdf2:sha256:260000:{salt}:{dk.hex()}"


def verify_password(password: str, stored: str) -> bool:
    try:
        _, algo, iters, salt, dk_hex = stored.split(":")
        dk = hashlib.pbkdf2_hmac(algo, password.encode(), salt.encode(), int(iters))
        return hmac.compare_digest(dk.hex(), dk_hex)
    except Exception:
        return False


# ── Token signing (HMAC-SHA256, no PyJWT dep) ────────────────────────────────

def _secret() -> bytes:
    return os.environ.get(
        "ORCA_AUTH_SECRET",
        "atheris-auth-dev-secret-CHANGE-THIS-IN-PRODUCTION"
    ).encode()


def create_token(payload: dict, expires_in: int = 86_400 * 30) -> str:
    p = {**payload, "exp": int(time.time()) + expires_in}
    data = (
        base64.urlsafe_b64encode(json.dumps(p, separators=(",", ":")).encode())
        .rstrip(b"=")
        .decode()
    )
    sig = hmac.new(_secret(), data.encode(), hashlib.sha256).hexdigest()
    return f"{data}.{sig}"


def verify_token(token: str) -> dict | None:
    try:
        data, sig = token.rsplit(".", 1)
        expected = hmac.new(_secret(), data.encode(), hashlib.sha256).hexdigest()
        if not hmac.compare_digest(sig, expected):
            return None
        padding = (4 - len(data) % 4) % 4
        payload = json.loads(base64.urlsafe_b64decode(data + "=" * padding))
        if payload.get("exp", 0) < time.time():
            return None
        return payload
    except Exception:
        return None
