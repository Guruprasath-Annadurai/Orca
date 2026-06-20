"""SMTP email delivery for Atheris auth flows.

Configure via env vars:
  SMTP_HOST, SMTP_PORT (default 587), SMTP_USER, SMTP_PASS
  SMTP_FROM  (default noreply@atheris.ai)
  APP_URL    (default http://localhost:7337)

If SMTP_HOST is not set, send_* functions return False (no-op).
"""
from __future__ import annotations

import os
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

SMTP_HOST = os.environ.get("SMTP_HOST", "")
SMTP_PORT = int(os.environ.get("SMTP_PORT", "587"))
SMTP_USER = os.environ.get("SMTP_USER", "")
SMTP_PASS = os.environ.get("SMTP_PASS", "")
SMTP_FROM = os.environ.get("SMTP_FROM", "noreply@atheris.ai")
APP_URL   = os.environ.get("APP_URL", "http://localhost:7337")

_STYLE = "font-family:monospace;background:#000;color:#e8e8e8;max-width:480px;margin:0 auto;padding:40px"
_BTN   = "display:inline-block;padding:12px 28px;background:#fff;color:#000;font-weight:700;letter-spacing:0.12em;text-decoration:none;font-family:monospace"
_MUTED = "color:#666;font-size:11px;margin-top:12px"


def is_configured() -> bool:
    return bool(SMTP_HOST and SMTP_USER and SMTP_PASS)


def send_email(to: str, subject: str, html: str, plain: str = "") -> bool:
    if not is_configured():
        return False
    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"]    = SMTP_FROM
    msg["To"]      = to
    if plain:
        msg.attach(MIMEText(plain, "plain"))
    msg.attach(MIMEText(html, "html"))
    try:
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=10) as s:
            s.ehlo()
            s.starttls()
            s.login(SMTP_USER, SMTP_PASS)
            s.sendmail(SMTP_FROM, [to], msg.as_string())
        return True
    except Exception:
        return False


def send_verification(to: str, token: str) -> bool:
    url  = f"{APP_URL}/api/auth/verify?token={token}"
    html = f"""
<div style="{_STYLE}">
  <p style="letter-spacing:0.35em;font-size:18px;color:#fff;margin:0 0 4px">ATHERIS</p>
  <p style="color:#555;letter-spacing:0.18em;font-size:10px;margin:0 0 28px">PRIVATE INTELLIGENCE</p>
  <p style="color:#999;letter-spacing:0.12em;margin:0 0 20px">VERIFY YOUR EMAIL</p>
  <p style="color:#ccc;margin:0 0 24px">Click the button below to activate your account. The link expires in 24 hours.</p>
  <a href="{url}" style="{_BTN}">VERIFY EMAIL</a>
  <p style="{_MUTED}">Or copy this URL:<br>{url}</p>
  <p style="{_MUTED}">If you didn't create an account, ignore this email.</p>
</div>"""
    return send_email(
        to, "Verify your Atheris account", html,
        plain=f"Verify your Atheris account:\n{url}\n\nLink expires in 24 hours.",
    )


def send_password_reset(to: str, token: str) -> bool:
    url  = f"{APP_URL}/?reset_token={token}"
    html = f"""
<div style="{_STYLE}">
  <p style="letter-spacing:0.35em;font-size:18px;color:#fff;margin:0 0 4px">ATHERIS</p>
  <p style="color:#555;letter-spacing:0.18em;font-size:10px;margin:0 0 28px">PRIVATE INTELLIGENCE</p>
  <p style="color:#999;letter-spacing:0.12em;margin:0 0 20px">PASSWORD RESET</p>
  <p style="color:#ccc;margin:0 0 24px">Click the button below to set a new password. The link expires in 1 hour.</p>
  <a href="{url}" style="{_BTN}">RESET PASSWORD</a>
  <p style="{_MUTED}">Or copy this URL:<br>{url}</p>
  <p style="{_MUTED}">If you didn't request a reset, ignore this email. Your password has not changed.</p>
</div>"""
    return send_email(
        to, "Reset your Atheris password", html,
        plain=f"Reset your Atheris password:\n{url}\n\nLink expires in 1 hour.",
    )
