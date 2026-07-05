"""
Citation compliance check — the actual mechanism behind "never hallucinate
sources," not just a prompt instruction hoping the model behaves.

Two situations, two different levels of enforcement:

  1. RAG context WAS available (a document is loaded for this session) and
     the persona's system prompt told it to cite sources as [D1], [D2], etc.
     This is MECHANICALLY CHECKABLE — if the response makes claims but uses
     zero citation markers despite context being provided, that's a
     detectable compliance failure, logged for governance visibility.

  2. No RAG context was available — the model is answering from its own
     training data, and there is no retrieval corpus to check citations
     against. This is NOT mechanically verifiable without an external fact
     database (out of scope here). The only lever is the system prompt
     instruction telling it to flag "this is from training, not verified" —
     weaker than case 1 by necessity, and this module says so honestly
     rather than pretending otherwise.
"""
from __future__ import annotations

import re

_CITATION_PATTERN = re.compile(r"\[D\d+\]")


def check_citations(response: str, context_block: str) -> dict:
    """
    Returns a compliance report:
      {
        "had_context": bool,       # was RAG context available for this turn
        "citations_used": [...],   # citation ids actually referenced, e.g. ["D1", "D2"]
        "compliant": bool,         # see rules below
        "note": str,               # human-readable explanation
      }

    Rules:
      - No context available -> always compliant (nothing to check against;
        the instruction to flag unverified training-knowledge claims is a
        system-prompt-level lever only, not mechanically verified here).
      - Context available, response used at least one [D#] marker -> compliant.
      - Context available, response used ZERO [D#] markers -> NOT compliant.
        This does not prove hallucination, but it means the response cannot
        be traced back to the retrieved sources — a real governance signal.
    """
    had_context = bool(context_block.strip())
    citations_used = sorted(set(_CITATION_PATTERN.findall(response)))

    if not had_context:
        return {
            "had_context": False,
            "citations_used": [],
            "compliant": True,
            "note": "No document context was available this turn — response relies on "
                    "training knowledge. Not mechanically verifiable; enforcement here is "
                    "limited to the system-prompt instruction to flag unverified claims.",
        }

    compliant = len(citations_used) > 0
    return {
        "had_context": True,
        "citations_used": citations_used,
        "compliant": compliant,
        "note": (
            f"Used {len(citations_used)} citation marker(s) from available document context."
            if compliant else
            "Document context WAS available but the response used zero [D#] citation "
            "markers — cannot be traced back to retrieved sources. Does not prove the "
            "answer is wrong, but it failed the citation-discipline check."
        ),
    }
