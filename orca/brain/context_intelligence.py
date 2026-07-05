"""
Context Intelligence — budget-aware conversation compression, replacing
blunt turn-count truncation with a keep/summarize policy.

Real bug this fixes: AgentLoop._history (orca/brain/agent.py) had NO size
cap at all — every turn appended forever. A long conversation would keep
growing the prompt sent to the model every single request until it blew
past the model's context window (num_ctx, typically 4096-8192 for the
variants this project targets), either erroring or getting silently
truncated in an uncontrolled way at the Ollama layer. ShortTermMemory
(orca/brain/memory.py) at least hard-truncates by turn count, but turn
count is a poor proxy for actual size — one one-word turn and one
2000-character turn cost very different amounts of context budget.

Policy, honestly scoped to what's implemented in this v1:
  KEEP      — recent turns, kept verbatim. Always wins over summarizing.
  SUMMARIZE — once the budget is exceeded, the OLDEST turns (everything
              before the keep-verbatim window) get compressed into one
              dense summary via a single LLM call, replacing many raw
              turns with one compact block.

Explicitly NOT implemented in this v1, stated plainly rather than
overclaiming a fuller framework than exists:
  FORGET         — dropping genuinely low-value turns (a bare "ok thanks")
                   before they even reach the summarize step. Would improve
                   summary density but adds a second LLM-judgment call per
                   compression cycle; descoped for now.
  RETRIEVE-LATER — pushing compressed-out content into long-term/vector
                   memory for future recall. Partially exists already via
                   MemoryEngine.commit_to_long_term(), but that's called
                   independently at the API layer (orca/serve/api.py), not
                   triggered BY this compression step. Wiring those together
                   is a real next step, not done here.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from orca.brain.providers import OrcaBrain

# Char-count budget, not a real tokenizer — same "approximate, don't add a
# dependency for something estimable" pattern as the rest of this project.
# ~4 chars/token is a standard rough English-text heuristic.
DEFAULT_BUDGET_CHARS = 12_000   # ≈3000 tokens — leaves headroom under a
                                # 4096-token num_ctx after system prompt + response
KEEP_RECENT_TURNS = 6           # last 6 messages (3 exchanges) always kept verbatim

_SUMMARY_PROMPT = """\
Summarize this conversation history into a dense, factual paragraph. Preserve:
- Specific facts, names, numbers, decisions made
- The user's stated goals or preferences
- Anything explicitly agreed or concluded

Omit small talk and filler. Be concise but do not lose information that would
matter for continuing this conversation coherently.

Conversation:
{transcript}

Summary:"""


@dataclass
class CompressionResult:
    history: list[dict]
    compressed: bool
    original_char_count: int
    new_char_count: int
    turns_summarized: int = 0


def _char_count(messages: list[dict]) -> int:
    return sum(len(m.get("content", "")) for m in messages)


def _summarize_via_llm(messages: list[dict], brain: "OrcaBrain") -> str:
    transcript = "\n".join(f"{m['role']}: {m['content']}" for m in messages)
    prompt = _SUMMARY_PROMPT.format(transcript=transcript[:8000])  # cap input to keep the summarization call itself bounded
    try:
        return brain.complete(
            [{"role": "user", "content": prompt}],
            temperature=0.2, max_tokens=400,
        ).strip()
    except Exception:
        return ""  # summarization failure handled by caller — falls back to hard truncation


def apply_context_policy(
    history: list[dict],
    brain: "OrcaBrain",
    budget_chars: int = DEFAULT_BUDGET_CHARS,
    keep_recent: int = KEEP_RECENT_TURNS,
) -> CompressionResult:
    """
    If `history` fits within budget_chars, returns it unchanged (cheap
    no-op — the common case for most conversations). Otherwise summarizes
    everything except the last `keep_recent` messages into one compact
    summary message, and returns [summary] + recent.
    """
    original_count = _char_count(history)

    if original_count <= budget_chars or len(history) <= keep_recent:
        return CompressionResult(history=history, compressed=False, original_char_count=original_count, new_char_count=original_count)

    older = history[:-keep_recent]
    recent = history[-keep_recent:]

    summary_text = _summarize_via_llm(older, brain)

    if not summary_text:
        # Summarization call failed (Ollama unreachable, etc.) — fall back
        # to hard truncation rather than blocking the conversation. Losing
        # the older context outright is worse than ideal but strictly
        # better than crashing or exceeding the context window.
        new_history = recent
    else:
        summary_message = {
            "role": "user",
            "content": f"[Summary of earlier conversation — {len(older)} messages compressed]\n{summary_text}",
        }
        new_history = [summary_message] + recent

    return CompressionResult(
        history=new_history,
        compressed=True,
        original_char_count=original_count,
        new_char_count=_char_count(new_history),
        turns_summarized=len(older),
    )
