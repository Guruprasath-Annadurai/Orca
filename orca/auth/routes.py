"""Auth API routes — signup, login, me, logout."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel

from orca.auth.crypto import create_token
from orca.auth.middleware import get_current_user
from orca.auth.store import (
    DAILY_LIMITS,
    User,
    authenticate,
    create_user,
    get_usage_today,
    get_user_by_email,
    set_user_tier,
    set_user_role,
    list_users,
)
from orca.auth.apikeys import create_key, list_keys, revoke_key
from orca.auth.rbac import require_permission
from orca.auth.tokens import make_verification_token, make_reset_token, verify_token as verify_auth_token
from orca.auth.email import send_verification, send_password_reset, is_configured as email_configured
from orca.serve import ratelimit

router = APIRouter(prefix="/api/auth", tags=["auth"])


class SignupRequest(BaseModel):
    email: str
    password: str
    name: str = ""


class LoginRequest(BaseModel):
    email: str
    password: str


def _make_response(user: User) -> dict:
    token  = create_token({"sub": user.id, "email": user.email, "tier": user.tier, "role": user.role})
    limits = DAILY_LIMITS.get(user.tier, DAILY_LIMITS["free"])
    usage  = get_usage_today(user.id)
    return {
        "token": token,
        "user": {
            "id":    user.id,
            "email": user.email,
            "name":  user.name,
            "tier":  user.tier,
            "role":  user.role,
        },
        "limits": limits,
        "usage":  usage,
    }


@router.post("/signup")
async def signup(req: SignupRequest, request: Request):
    ratelimit.enforce(request, ratelimit.AUTH_SIGNUP)
    if len(req.password) < 8:
        raise HTTPException(400, "Password must be at least 8 characters")
    if "@" not in req.email or "." not in req.email.split("@")[-1]:
        raise HTTPException(400, "Invalid email address")
    if get_user_by_email(req.email):
        raise HTTPException(409, "An account with this email already exists")
    try:
        user = create_user(req.email, req.password, req.name)
    except Exception as e:
        raise HTTPException(500, f"Could not create account: {e}")
    # Send verification email (non-blocking best-effort)
    if email_configured():
        token = make_verification_token(user.id, user.email)
        send_verification(user.email, token)
    return _make_response(user)


@router.post("/login")
async def login(req: LoginRequest, request: Request):
    ratelimit.enforce(request, ratelimit.AUTH_LOGIN)
    user = authenticate(req.email, req.password)
    if not user:
        raise HTTPException(401, "Invalid email or password")
    return _make_response(user)


@router.get("/me")
async def me(user: User = Depends(get_current_user)):
    limits = DAILY_LIMITS.get(user.tier, DAILY_LIMITS["free"])
    usage  = get_usage_today(user.id)
    return {
        "user":   {"id": user.id, "email": user.email, "name": user.name, "tier": user.tier, "role": user.role},
        "limits": limits,
        "usage":  usage,
    }


# ── Email / password routes ───────────────────────────────────────────────────

class ForgotRequest(BaseModel):
    email: str


class ResetRequest(BaseModel):
    token: str
    password: str


class ChangePasswordRequest(BaseModel):
    current_password: str
    new_password: str


@router.get("/email-status")
async def email_status():
    return {"configured": email_configured()}


@router.post("/signup/resend-verification")
async def resend_verification(user: User = Depends(get_current_user)):
    if user.verified:
        raise HTTPException(400, "Account already verified")
    token = make_verification_token(user.id, user.email)
    sent  = send_verification(user.email, token)
    return {"sent": sent, "email_configured": email_configured()}


@router.get("/verify")
async def verify_email(token: str):
    payload = verify_auth_token(token, "verify")
    if not payload:
        from fastapi.responses import HTMLResponse
        return HTMLResponse("<h2>Verification link is invalid or expired.</h2>", status_code=400)
    from orca.auth.store import mark_verified, get_user_by_id
    mark_verified(payload["sub"])
    return HTMLResponse("""<!DOCTYPE html><html><head><meta charset="UTF-8">
<style>body{background:#000;color:#e8e8e8;font-family:monospace;display:flex;align-items:center;
justify-content:center;height:100vh;margin:0}div{text-align:center}</style></head><body>
<div><p style="letter-spacing:.35em;font-size:20px;color:#fff">ATHERIS</p>
<p style="color:#888;letter-spacing:.15em;margin-top:8px">EMAIL VERIFIED</p>
<p style="color:#555;margin-top:20px">Your account is now active. <a href="/" style="color:#fff">Return to Atheris →</a></p>
</div></body></html>""")


@router.post("/forgot-password")
async def forgot_password(req: ForgotRequest, request: Request):
    ratelimit.enforce(request, ratelimit.AUTH_FORGOT_PW)
    user = get_user_by_email(req.email)
    if user:
        token = make_reset_token(user.id, user.email)
        send_password_reset(user.email, token)
    # Always return 200 — never reveal if email exists
    return {"sent": True}


@router.post("/reset-password")
async def reset_password(req: ResetRequest):
    if len(req.password) < 8:
        raise HTTPException(400, "Password must be at least 8 characters")
    payload = verify_auth_token(req.token, "reset")
    if not payload:
        raise HTTPException(400, "Reset link is invalid or expired")
    from orca.auth.store import update_password, get_user_by_id
    update_password(payload["sub"], req.password)
    user = get_user_by_id(payload["sub"])
    if not user:
        raise HTTPException(404, "User not found")
    return _make_response(user)


@router.post("/change-password")
async def change_password(req: ChangePasswordRequest, user: User = Depends(get_current_user)):
    from orca.auth.store import authenticate, update_password
    if not authenticate(user.email, req.current_password):
        raise HTTPException(400, "Current password is incorrect")
    if len(req.new_password) < 8:
        raise HTTPException(400, "New password must be at least 8 characters")
    update_password(user.id, req.new_password)
    return {"updated": True}


# ── API Key routes ────────────────────────────────────────────────────────────

class ApiKeyRequest(BaseModel):
    name: str = ""


@router.post("/apikeys")
async def create_api_key(
    req: ApiKeyRequest,
    user: User = Depends(require_permission("api_keys")),
):
    kid, raw = create_key(user.id, req.name)
    return {"key_id": kid, "key": raw, "name": req.name}


@router.get("/apikeys")
async def get_api_keys(user: User = Depends(require_permission("api_keys"))):
    return {"keys": list_keys(user.id)}


@router.delete("/apikeys/{key_id}")
async def delete_api_key(key_id: str, user: User = Depends(require_permission("api_keys"))):
    ok = revoke_key(key_id, user.id)
    if not ok:
        raise HTTPException(404, "Key not found")
    return {"revoked": True}


# ── Admin routes ──────────────────────────────────────────────────────────────

class TierUpdate(BaseModel):
    tier: str


class RoleUpdate(BaseModel):
    role: str


@router.get("/admin/users")
async def admin_list_users(
    limit: int = 50,
    offset: int = 0,
    admin: User = Depends(require_permission("manage_users")),
):
    return {"users": list_users(limit=limit, offset=offset)}


@router.patch("/admin/users/{user_id}/tier")
async def admin_set_tier(
    user_id: str,
    body: TierUpdate,
    admin: User = Depends(require_permission("manage_users")),
):
    if body.tier not in ("free", "pro", "enterprise"):
        raise HTTPException(400, "Invalid tier")
    set_user_tier(user_id, body.tier)
    return {"updated": True}


@router.patch("/admin/users/{user_id}/role")
async def admin_set_role(
    user_id: str,
    body: RoleUpdate,
    admin: User = Depends(require_permission("manage_users")),
):
    if body.role not in ("owner", "admin", "member", "viewer"):
        raise HTTPException(400, "Invalid role")
    set_user_role(user_id, body.role)
    return {"updated": True}
