"""
License delivery via email.

Required env vars (set in .env):
  SMTP_HOST   — e.g. smtp.gmail.com
  SMTP_PORT   — e.g. 587
  SMTP_USER   — your sending email
  SMTP_PASS   — app password or SMTP password
  SMTP_FROM   — From address (defaults to SMTP_USER)

If SMTP is not configured, the function logs to stdout and returns False.
"""
from __future__ import annotations

import os
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText


def send_license_email(
    to_email: str,
    license_key: str,
    tier: str,
    seats: int,
    days: int,
) -> bool:
    """
    Email a license key to the customer.
    Returns True on success, False if SMTP not configured or send failed.
    """
    host     = os.environ.get("SMTP_HOST", "")
    port     = int(os.environ.get("SMTP_PORT", "587"))
    user     = os.environ.get("SMTP_USER", "")
    password = os.environ.get("SMTP_PASS", "")
    from_addr = os.environ.get("SMTP_FROM", user)

    if not host or not user:
        # Email not configured — log to stdout so admin can see the key
        print(f"\n[LICENSE KEY GENERATED — EMAIL NOT CONFIGURED]")
        print(f"  To:   {to_email}")
        print(f"  Key:  {license_key}")
        print(f"  Tier: {tier}  Seats: {seats}  Days: {days}")
        return False

    expiry_note = f"{days} days" if days > 0 else "lifetime (never expires)"

    msg = MIMEMultipart("alternative")
    msg["Subject"] = f"Your Orca {tier.title()} License Key"
    msg["From"]    = from_addr
    msg["To"]      = to_email

    plain = f"""\
Thank you for purchasing Orca {tier.title()}!

Your license key:

  {license_key}

Activate it in your terminal:
  orca activate {license_key}

Details:
  Tier:   {tier.title()}
  Seats:  {seats}
  Valid:  {expiry_note}

To check your license at any time:
  orca license

Questions? Reply to this email.

— The Orca Team
Your AI. Your hardware. Your data.
"""

    html = f"""\
<!DOCTYPE html>
<html>
<body style="margin:0;padding:0;background:#050505;font-family:monospace;">
<div style="max-width:560px;margin:40px auto;padding:40px 48px;background:#0a0a0a;border:1px solid #1f1f1f;">

  <div style="font-size:22px;font-weight:700;letter-spacing:0.15em;color:#ffffff;margin-bottom:6px;">
    ORCA
  </div>
  <div style="font-size:10px;letter-spacing:0.2em;color:#444;margin-bottom:36px;">
    APEX INTELLIGENCE
  </div>

  <div style="font-size:13px;color:#999;margin-bottom:28px;">
    Thank you for purchasing <strong style="color:#fff;">Orca {tier.title()}</strong>.
    Your license key is below.
  </div>

  <!-- Key block -->
  <div style="background:#111;border:1px solid #2a2a2a;padding:20px 24px;margin-bottom:28px;">
    <div style="font-size:9px;letter-spacing:0.25em;color:#555;margin-bottom:10px;">
      YOUR LICENSE KEY
    </div>
    <div style="font-size:15px;letter-spacing:0.06em;color:#ffffff;word-break:break-all;">
      {license_key}
    </div>
  </div>

  <!-- Activate command -->
  <div style="font-size:11px;color:#666;margin-bottom:8px;letter-spacing:0.1em;">
    ACTIVATE IN YOUR TERMINAL
  </div>
  <div style="background:#111;border:1px solid #1f1f1f;padding:12px 16px;margin-bottom:28px;">
    <span style="color:#00cc66;font-size:12px;">
      orca activate {license_key}
    </span>
  </div>

  <!-- Details table -->
  <table style="width:100%;font-size:11px;border-collapse:collapse;margin-bottom:32px;">
    <tr>
      <td style="padding:5px 0;color:#555;letter-spacing:0.1em;">TIER</td>
      <td style="color:#fff;">{tier.upper()}</td>
    </tr>
    <tr>
      <td style="padding:5px 0;color:#555;letter-spacing:0.1em;">SEATS</td>
      <td style="color:#fff;">{seats}</td>
    </tr>
    <tr>
      <td style="padding:5px 0;color:#555;letter-spacing:0.1em;">VALID</td>
      <td style="color:#fff;">{expiry_note}</td>
    </tr>
  </table>

  <div style="font-size:10px;color:#333;letter-spacing:0.05em;border-top:1px solid #1f1f1f;padding-top:20px;">
    Questions? Reply to this email.<br>
    Your AI. Your hardware. Your data.
  </div>

</div>
</body>
</html>
"""

    msg.attach(MIMEText(plain, "plain"))
    msg.attach(MIMEText(html, "html"))

    try:
        with smtplib.SMTP(host, port, timeout=15) as smtp:
            smtp.ehlo()
            smtp.starttls()
            smtp.login(user, password)
            smtp.sendmail(from_addr, [to_email], msg.as_string())
        return True
    except Exception as e:
        print(f"[LICENSE EMAIL FAILED] {to_email}: {e}")
        print(f"  Key: {license_key}")
        return False
