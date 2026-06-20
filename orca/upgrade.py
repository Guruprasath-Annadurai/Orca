"""
Orca self-updater — checks PyPI for a newer version and upgrades in-place.
"""
from __future__ import annotations

import subprocess
import sys
from typing import Optional

_PYPI_URL = "https://pypi.org/pypi/orca-ai/json"
_PACKAGE   = "orca-ai"


def _current_version() -> str:
    try:
        from importlib.metadata import version
        return version(_PACKAGE)
    except Exception:
        from orca.__version__ import __version__
        return __version__


def get_latest_version() -> Optional[str]:
    """Return the latest version string from PyPI, or None on network failure."""
    try:
        import urllib.request, json
        with urllib.request.urlopen(_PYPI_URL, timeout=8) as r:
            data = json.loads(r.read())
        return data["info"]["version"]
    except Exception:
        return None


def _parse_version(v: str) -> tuple[int, ...]:
    try:
        return tuple(int(x) for x in v.split(".")[:3])
    except Exception:
        return (0,)


def is_update_available() -> tuple[bool, str, str]:
    """
    Returns (available, current, latest).
    available=False if network is down or already up to date.
    """
    current = _current_version()
    latest  = get_latest_version()
    if not latest:
        return False, current, ""
    if _parse_version(latest) > _parse_version(current):
        return True, current, latest
    return False, current, latest


def self_update(yes: bool = False) -> bool:
    """
    Upgrade orca-ai via pip. Returns True on success.
    Raises RuntimeError if pip fails.
    """
    available, current, latest = is_update_available()
    if not available:
        return False

    cmd = [sys.executable, "-m", "pip", "install", "--upgrade", _PACKAGE, "--quiet"]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "pip upgrade failed")
    return True
