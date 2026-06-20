"""
Orca License — key generation, validation, gating, Stripe fulfillment.

Quick reference:
  from orca.license import has_feature, gate, current_tier
  from orca.license import get_active_license, generate_key, validate_key
"""
from orca.license.keys  import generate_key, validate_key, LicenseKey, TIER_FEATURES
from orca.license.store import get_active_license, current_tier, save_license, clear_license, activation_email
from orca.license.gate  import has_feature, gate, require_license

__all__ = [
    "generate_key",
    "validate_key",
    "LicenseKey",
    "TIER_FEATURES",
    "get_active_license",
    "current_tier",
    "save_license",
    "clear_license",
    "activation_email",
    "has_feature",
    "gate",
    "require_license",
]
