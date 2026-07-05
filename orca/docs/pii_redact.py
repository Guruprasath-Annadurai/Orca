"""
PII redaction on document ingest — scrubs common sensitive-data patterns
from uploaded documents BEFORE chunking/embedding into the RAG vector store.

Scope, stated honestly:
  - Applies to DOCUMENT UPLOADS ONLY (orca/serve/api.py's /api/docs/upload),
    not live chat messages. A user typing their own email/phone in a chat
    message is doing so intentionally as conversational context — silently
    mangling that would be worse than the risk it prevents. The real risk
    this addresses is someone uploading a document (a resume, an exported
    spreadsheet, a support ticket) that contains a THIRD PARTY's sensitive
    data they didn't think about before it gets embedded into a persistent
    vector store.
  - Pattern-based, not a trained PII-detection model — same honesty
    posture as orca/train/redteam.py: a floor, not a ceiling. Catches
    common, well-structured patterns (SSN, major-card-brand credit card
    numbers validated via Luhn, email, US phone numbers). Misses anything
    that doesn't match a known structural pattern (a name alone, a home
    address, a non-US ID number).
  - Credit card numbers are Luhn-checksum validated before redaction to
    avoid false-positiving on ordinary 16-digit numbers that happen to
    appear in a document (order IDs, tracking numbers) but aren't valid
    card numbers.
  - SSN matching is deliberately PURE STRUCTURE (\\d{3}-\\d{2}-\\d{4}), not
    narrowed by "is this a government-issued-valid range" rules (e.g.
    excluding area numbers 900-999). An earlier version tried to be clever
    about that and it backfired: a test/example SSN in the 900s range
    silently passed through unredacted because it "wasn't a real SSN" by
    that narrower rule — but it was still SSN-SHAPED data a document reader
    would treat as sensitive. For a redaction tool, over-redacting a stray
    9-digit-in-3-2-4-format number is a low-cost false positive; under-
    redacting something SSN-shaped is a real leak. Erring toward the former.
"""
from __future__ import annotations

import re

_SSN_PATTERN = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
_EMAIL_PATTERN = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
_PHONE_PATTERN = re.compile(
    # (?<!\d)/(?!\d) instead of \b: \b behaves oddly around a literal "(" —
    # it's a non-word char, so \b actually matches BETWEEN "(" and the digit
    # after it, not before the "(" itself, causing the opening paren to be
    # left outside the match (dangling, unredacted) when \b was used here.
    # Digit-lookaround also prevents matching mid-way through a longer digit
    # run (e.g. inside a credit card number) — a plain \d{3}... without any
    # anchor would happily match the middle 10 digits of a 16-digit number.
    r"(?<!\d)(?:\+?1[-.\s]?)?(?:\(\d{3}\)|\d{3})[-.\s]?\d{3}[-.\s]?\d{4}(?!\d)"
)
_CC_CANDIDATE_PATTERN = re.compile(r"(?<!\d)(?:\d[ -]?){13,19}(?!\d)")


def _luhn_valid(digits: str) -> bool:
    """Standard Luhn checksum — same algorithm every card issuer uses to validate card numbers."""
    total = 0
    reverse_digits = digits[::-1]
    for i, ch in enumerate(reverse_digits):
        d = int(ch)
        if i % 2 == 1:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


def _redact_credit_cards(text: str) -> tuple[str, int]:
    count = 0

    def _replace(match: re.Match) -> str:
        nonlocal count
        raw = match.group()
        digits = re.sub(r"[ -]", "", raw)
        if 13 <= len(digits) <= 19 and digits.isdigit() and _luhn_valid(digits):
            count += 1
            return "[REDACTED-CC]"
        return raw  # not a valid card number by Luhn — leave it alone, likely a false positive candidate

    redacted = _CC_CANDIDATE_PATTERN.sub(_replace, text)
    return redacted, count


def redact_pii(text: str) -> tuple[str, dict]:
    """
    Returns (redacted_text, report). report is counts only — never the
    actual sensitive values — safe to pass straight to audit.log for
    governance visibility without re-exposing what was redacted.
    """
    report = {"ssn": 0, "email": 0, "phone": 0, "credit_card": 0}

    text, ssn_count = _SSN_PATTERN.subn("[REDACTED-SSN]", text)
    report["ssn"] = ssn_count

    text, email_count = _EMAIL_PATTERN.subn("[REDACTED-EMAIL]", text)
    report["email"] = email_count

    # Credit cards BEFORE phone — a longer digit run (13-19 digits) must be
    # consumed as a whole before the shorter phone pattern gets a chance to
    # match a 10-digit slice out of its middle, which would corrupt the
    # run and make the Luhn check on the remainder meaningless.
    text, cc_count = _redact_credit_cards(text)
    report["credit_card"] = cc_count

    text, phone_count = _PHONE_PATTERN.subn("[REDACTED-PHONE]", text)
    report["phone"] = phone_count

    report["total"] = sum(report.values())
    return text, report
