"""
TOTP (RFC 6238) — time-based one-time passwords for 2FA.

Implemented natively with stdlib hmac/hashlib rather than adding a pyotp
dependency — the algorithm is ~30 lines and this project already implements
its own crypto primitives elsewhere (orca/audit.py's hash chain, orca/auth/crypto.py's
password hashing) rather than reaching for a library for something this small.

Standard TOTP parameters: SHA-1, 30-second time step, 6-digit codes — these
are the universal defaults every authenticator app (Google Authenticator,
Authy, 1Password, etc.) assumes. Deviating from them would break
compatibility with real authenticator apps for no benefit.
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import os
import struct
import time
import urllib.parse

_TIME_STEP = 30
_DIGITS = 6


def generate_totp_secret() -> str:
    """Random base32 secret, the format every authenticator app expects."""
    raw = os.urandom(20)  # 160 bits — standard TOTP secret length
    return base64.b32encode(raw).decode("utf-8").rstrip("=")


def _hotp(secret_b32: str, counter: int) -> str:
    """HOTP (RFC 4226) — TOTP is HOTP with counter = floor(time / time_step)."""
    # Restore base32 padding stripped by generate_totp_secret()
    padded = secret_b32 + "=" * ((8 - len(secret_b32) % 8) % 8)
    key = base64.b32decode(padded.upper())
    msg = struct.pack(">Q", counter)
    digest = hmac.new(key, msg, hashlib.sha1).digest()
    offset = digest[-1] & 0x0F
    code_int = (struct.unpack(">I", digest[offset:offset + 4])[0] & 0x7FFFFFFF) % (10 ** _DIGITS)
    return str(code_int).zfill(_DIGITS)


def totp_now(secret_b32: str, at_time: float | None = None) -> str:
    t = at_time if at_time is not None else time.time()
    counter = int(t // _TIME_STEP)
    return _hotp(secret_b32, counter)


def verify_totp(secret_b32: str, code: str, window: int = 1, at_time: float | None = None) -> bool:
    """
    Verify a submitted code against the current time step, allowing ±window
    steps of clock drift (default ±30s) — phones and servers aren't always
    in perfect sync, and rejecting a code that's 1 step off due to drift
    would make 2FA unusable, not more secure.
    """
    if not code or not code.isdigit() or len(code) != _DIGITS:
        return False

    t = at_time if at_time is not None else time.time()
    current_counter = int(t // _TIME_STEP)

    for offset in range(-window, window + 1):
        expected = _hotp(secret_b32, current_counter + offset)
        if hmac.compare_digest(expected, code):
            return True
    return False


def provisioning_uri(secret_b32: str, account_name: str, issuer: str = "Orca") -> str:
    """
    otpauth:// URI — scan-compatible with every major authenticator app.
    No QR image generation server-side (avoids an image-lib dependency for
    something the frontend can render client-side, or the user can enter
    the secret manually — every authenticator app supports both).
    """
    label = urllib.parse.quote(f"{issuer}:{account_name}")
    params = urllib.parse.urlencode({
        "secret": secret_b32, "issuer": issuer, "algorithm": "SHA1",
        "digits": str(_DIGITS), "period": str(_TIME_STEP),
    })
    return f"otpauth://totp/{label}?{params}"
