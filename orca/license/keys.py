"""
Orca license key generation and offline validation.

Key format:
  ORCA-{TIER}-{B32_GROUP}-{B32_GROUP}-{B32_GROUP}-{B32_GROUP}-{B32_GROUP}

Payload (15 bytes): version(1) + tier(1) + seats(1) + expiry_days(4) + nonce(8)
Signature (10 bytes): HMAC-SHA256(secret, payload)[:10]
Total encoded: 25 bytes → ~40 base32 chars → 8 groups of 5

Offline-verifiable — no network needed. The same ORCA_LICENSE_SECRET
used to generate a key must be present to verify it. Distribute the
app with a secret baked in, or set the env var at build time.
"""
from __future__ import annotations

import hashlib
import hmac as _hmac
import os
import secrets
import struct
from base64 import b32decode, b32encode
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

# ─── Tier registry ─────────────────────────────────────────────────────────────

TIER_INT: dict[str, int] = {
    "free":       0,
    "pro":        1,
    "enterprise": 2,
}
INT_TIER: dict[int, str] = {v: k for k, v in TIER_INT.items()}

TIER_LABEL: dict[str, str] = {
    "free":       "FREE",
    "pro":        "PRO",
    "enterprise": "ENT",
}
LABEL_TIER: dict[str, str] = {v: k for k, v in TIER_LABEL.items()}

# Features unlocked per tier (cumulative)
TIER_FEATURES: dict[str, set[str]] = {
    "free": {
        "core_chat",
        "data_seed",
        "web_basic",
        "doctor",
        "status",
    },
    "pro": {
        "core_chat",
        "data_seed",
        "web_basic",
        "doctor",
        "status",
        "ultra",            # Multi-agent Ultra orchestration
        "cloud_train",      # SSH cloud GPU training
        "web_ultra",        # Ultra mode in browser UI
        "memory_advanced",  # ChromaDB vector memory
        "serve",            # Web server
    },
    "enterprise": {"*"},    # All features, unlimited seats
}

# ─── Secret ────────────────────────────────────────────────────────────────────

# IMPORTANT: Change this before distributing. Set ORCA_LICENSE_SECRET in your
# build environment or .env file. This value is used for HMAC signing.
_BUILT_IN_SECRET = (
    "orca-lk-v1-4a7f91bc-e8d2-4c5e-b3f1-"
    "9a2c8e4d7b6f-change-before-shipping"
)


def _secret() -> bytes:
    return os.environ.get("ORCA_LICENSE_SECRET", _BUILT_IN_SECRET).encode()


# ─── Key dataclass ─────────────────────────────────────────────────────────────

@dataclass
class LicenseKey:
    raw: str                       # original key string as entered
    tier: str                      # "free" | "pro" | "enterprise"
    seats: int
    expiry: datetime | None        # None = lifetime
    valid: bool
    error: str = ""

    @property
    def is_lifetime(self) -> bool:
        return self.expiry is None

    @property
    def days_remaining(self) -> int | None:
        if self.expiry is None:
            return None
        delta = self.expiry - datetime.now(tz=timezone.utc)
        return max(0, delta.days)

    def has_feature(self, feature: str) -> bool:
        allowed = TIER_FEATURES.get(self.tier, set())
        return "*" in allowed or feature in allowed


# ─── Generation ────────────────────────────────────────────────────────────────

def generate_key(
    tier: str = "pro",
    seats: int = 1,
    days: int = 365,
) -> str:
    """
    Generate a signed license key.
    days=0  → lifetime (never expires)
    seats   → max concurrent seats (1-255)
    """
    if tier not in TIER_INT:
        raise ValueError(f"Unknown tier '{tier}'. Valid: {list(TIER_INT)}")

    seats = max(1, min(seats, 255))
    tier_byte = TIER_INT[tier]

    if days == 0:
        expiry_epoch_days = 0  # sentinel for lifetime
    else:
        expiry_dt = datetime.now(tz=timezone.utc) + timedelta(days=days)
        expiry_epoch_days = int(expiry_dt.timestamp()) // 86400

    nonce = secrets.token_bytes(8)

    # version=1, tier, seats, expiry (4 bytes, big-endian unsigned int), nonce (8 bytes)
    payload = struct.pack(">BBBI", 1, tier_byte, seats, expiry_epoch_days) + nonce

    sig = _hmac.new(_secret(), payload, hashlib.sha256).digest()[:10]

    encoded = b32encode(payload + sig).decode("ascii").rstrip("=")

    # Split into groups of 5 for readability
    groups = [encoded[i : i + 5] for i in range(0, len(encoded), 5)]

    label = TIER_LABEL[tier]
    return f"ORCA-{label}-" + "-".join(groups)


# ─── Validation ────────────────────────────────────────────────────────────────

def validate_key(raw: str) -> LicenseKey:
    """Validate a license key offline. Returns LicenseKey with valid=True/False."""

    def _fail(msg: str) -> LicenseKey:
        return LicenseKey(raw=raw, tier="free", seats=0, expiry=None, valid=False, error=msg)

    key = raw.strip().upper().replace(" ", "")
    parts = key.split("-")

    # Format: ORCA-{LABEL}-{groups...}
    if len(parts) < 4 or parts[0] != "ORCA":
        return _fail("Invalid key format (must start with ORCA-)")

    label = parts[1]
    tier = LABEL_TIER.get(label)
    if tier is None:
        return _fail(f"Unknown tier label '{label}'")

    body = "".join(parts[2:])
    padding = (8 - len(body) % 8) % 8
    try:
        full = b32decode(body + "=" * padding)
    except Exception:
        return _fail("Key contains invalid characters")

    # Expect 15 payload bytes + 10 sig bytes = 25 total
    if len(full) < 25:
        return _fail("Key is too short")

    payload, sig = full[:15], full[15:25]

    expected_sig = _hmac.new(_secret(), payload, hashlib.sha256).digest()[:10]
    if not _hmac.compare_digest(sig, expected_sig):
        return _fail("Key signature is invalid")

    version, tier_byte, seats, expiry_epoch_days = struct.unpack(">BBBI", payload[:7])

    if version != 1:
        return _fail(f"Unknown key version {version}")

    payload_tier = INT_TIER.get(tier_byte)
    if payload_tier != tier:
        return _fail("Tier label does not match key payload")

    # Expiry
    if expiry_epoch_days == 0:
        expiry_dt = None
        expired = False
    else:
        expiry_dt = datetime.fromtimestamp(expiry_epoch_days * 86400, tz=timezone.utc)
        expired = datetime.now(tz=timezone.utc) > expiry_dt

    if expired:
        return LicenseKey(
            raw=raw, tier=tier, seats=seats, expiry=expiry_dt,
            valid=False, error=f"License expired on {expiry_dt.strftime('%Y-%m-%d')}",
        )

    return LicenseKey(raw=raw, tier=tier, seats=seats, expiry=expiry_dt, valid=True)


# ─── Helpers ───────────────────────────────────────────────────────────────────

def format_expiry(lk: LicenseKey) -> str:
    if lk.expiry is None:
        return "lifetime"
    remaining = lk.days_remaining
    if remaining == 0:
        return "expires today"
    if remaining <= 30:
        return f"expires in {remaining} days ({lk.expiry.strftime('%Y-%m-%d')})"
    return lk.expiry.strftime("%Y-%m-%d")
