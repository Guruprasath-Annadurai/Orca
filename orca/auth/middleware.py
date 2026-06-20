"""FastAPI auth dependency — extracts and validates the current user."""
from __future__ import annotations

from typing import Optional

from fastapi import Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from orca.auth.crypto import verify_token
from orca.auth.store import User, get_user_by_id
from orca.auth.apikeys import verify_api_key

_bearer = HTTPBearer(auto_error=False)


async def get_current_user(
    creds: Optional[HTTPAuthorizationCredentials] = Depends(_bearer),
) -> User:
    if not creds:
        raise HTTPException(status_code=401, detail="Not authenticated")
    token = creds.credentials
    # API key path (athr_ prefix)
    if token.startswith("athr_"):
        uid = verify_api_key(token)
        if not uid:
            raise HTTPException(status_code=401, detail="Invalid or revoked API key")
        user = get_user_by_id(uid)
        if not user:
            raise HTTPException(status_code=401, detail="User not found")
        return user
    # JWT path
    payload = verify_token(token)
    if not payload:
        raise HTTPException(status_code=401, detail="Invalid or expired token")
    user = get_user_by_id(payload.get("sub", ""))
    if not user:
        raise HTTPException(status_code=401, detail="User not found")
    return user


async def get_current_user_optional(
    creds: Optional[HTTPAuthorizationCredentials] = Depends(_bearer),
) -> Optional[User]:
    if not creds:
        return None
    try:
        return await get_current_user(creds)
    except HTTPException:
        return None
