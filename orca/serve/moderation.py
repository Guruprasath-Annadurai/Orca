"""
Input moderation — checks user messages BEFORE they reach the model.

Same honesty pattern as orca/train/redteam.py: keyword/pattern-based, a
floor not a ceiling. A real production deployment handling adversarial
traffic at scale should layer a trained moderation classifier on top of
this — this module says so explicitly rather than overselling itself as
comprehensive.

Three distinct actions, not one binary block/allow — because the right
response to different categories is genuinely different:

  BLOCK        — hard refusal, logged, generation never happens. Reserved
                 for the narrow set of categories where there is no
                 legitimate framing: CSAM-adjacent requests, and requests
                 for operational synthesis/construction instructions for
                 mass-casualty weapons (bio/chem/nuclear/radiological,
                 or viable explosive devices).

  SUPPORT       — self-harm / suicide ideation. Deliberately NEVER blocked.
                 Blocking or refusing someone expressing suicidal ideation
                 is not a safety win — it's the opposite of what mental
                 health crisis response guidance recommends. Instead: flag
                 for governance visibility and inject crisis resources into
                 context so the model's response includes them, but let
                 generation proceed normally.

  FLAG          — harassment / hate-speech-adjacent language. Logged for
                 governance visibility (same as red-team bias flags — a
                 signal for review, not proof of intent), generation
                 proceeds. Hard-blocking this category over-blocks
                 legitimate uses (academic discussion, quoting for
                 critique, content moderation research itself).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

# ─────────────────────────────────────────────────────────────────────────────
#  BLOCK — no legitimate framing exists for these
# ─────────────────────────────────────────────────────────────────────────────

_BLOCK_PATTERNS = [
    # CSAM-adjacent — sparse and deliberately narrow to avoid false-positiving
    # on legitimate child-safety/protection discussion; a real deployment
    # MUST layer a dedicated, specialized classifier here, not rely on this.
    r"\bsexual.{0,20}\b(child|minor|underage)\b",
    r"\b(child|minor|underage)\b.{0,20}\bsexual\b",
    # Mass-casualty weapon synthesis/construction instructions
    r"\bhow to (synthesize|make|build|produce)\b.{0,30}\b(nerve agent|sarin|VX|anthrax|"
    r"ricin|bioweapon|chemical weapon|dirty bomb|nuclear device)\b",
    r"\bstep.by.step\b.{0,30}\b(explosive device|pipe bomb|detonator)\b.{0,30}\b(build|construct|assemble)\b",
]

# ─────────────────────────────────────────────────────────────────────────────
#  SUPPORT — self-harm / suicide ideation, never blocked
# ─────────────────────────────────────────────────────────────────────────────

_SUPPORT_PATTERNS = [
    r"\bwant to (die|kill myself|end (it|my life))\b",
    r"\bsuicid\w*\b",
    r"\bself.harm\b", r"\bhurt(ing)? myself\b",
    r"\bno reason to (live|go on)\b",
    r"\bbetter off (dead|without me)\b",
]

CRISIS_RESOURCES = (
    "If you're in crisis or thinking about suicide, please reach out for support:\n"
    "- US: 988 Suicide & Crisis Lifeline (call or text 988)\n"
    "- International: https://findahelpline.com\n"
    "You deserve support, and talking to someone now can help."
)

# ─────────────────────────────────────────────────────────────────────────────
#  FLAG — harassment / hate-speech-adjacent, logged not blocked
# ─────────────────────────────────────────────────────────────────────────────

_FLAG_PATTERNS = [
    r"\bwrite (a|an|some) (insult|hateful|degrading)\b.{0,30}\b(about|targeting)\b",
    r"\bgenerate (hate speech|slurs)\b",
    r"\bmake fun of\b.{0,20}\b(disab|race|religion|ethnic)\w*\b",
]


@dataclass
class ModerationResult:
    action: str  # "allow" | "block" | "support" | "flag"
    flagged_categories: list[str] = field(default_factory=list)
    matched_pattern: str = ""


def check_input(text: str) -> ModerationResult:
    """
    Checks a single input message. BLOCK takes precedence over SUPPORT over
    FLAG over allow — if a message somehow matches both a block pattern and
    a support pattern, treat it as the more serious category.
    """
    lowered = text.lower()

    for pattern in _BLOCK_PATTERNS:
        if re.search(pattern, lowered):
            return ModerationResult(action="block", flagged_categories=["hard_block"], matched_pattern=pattern)

    for pattern in _SUPPORT_PATTERNS:
        if re.search(pattern, lowered):
            return ModerationResult(action="support", flagged_categories=["self_harm"], matched_pattern=pattern)

    for pattern in _FLAG_PATTERNS:
        if re.search(pattern, lowered):
            return ModerationResult(action="flag", flagged_categories=["harassment"], matched_pattern=pattern)

    return ModerationResult(action="allow")
