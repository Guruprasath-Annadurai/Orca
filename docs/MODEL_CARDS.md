# Model Cards & the Persona Claim Gate

Every Orca model variant ships a signed model card documenting its actual,
measured accuracy and safety scores — not marketing claims. This document
explains how to read one and how the enforcement mechanism behind it works.

## Generating a card

```bash
orca train eval --ollama orca-core --ci        # accuracy/style eval
orca train redteam --model orca-core --ci      # safety/jailbreak/bias/calibration
orca train card core                            # generates the signed card from the two reports above
orca train cards                                # lists all generated cards
```

A card is *not* regenerated automatically — it reflects whatever eval/red-team reports exist on disk at the moment you run `orca train card`. Rerun eval/redteam first if you want the card to reflect a new model version.

## What's in a card

- `training_data_summary` — example count, base model, domain coverage
- `eval_scores` — accuracy (keyword-coverage on a 50-prompt golden set), style score (LLM-as-judge), speed (tokens/sec)
- `safety_scores` — jailbreak block rate, bias flag rate, toxicity flags, calibration score (does the model correctly push back on false premises)
- `known_limitations` — **derived from the actual numbers above, not boilerplate.** If accuracy is below 70%, the card says so with the exact percentage. If jailbreak block rate is below 95%, it's listed with the exact rate.
- `persona_claim_approved` / `persona_claim_reason` — see below
- `signature` — HMAC signature over the card contents; `verify_model_card()` detects if a card was edited after generation

## The persona claim gate

Orca ships three persona system prompts — Genesis (nano), Novus (core),
Aeternum (ultra) — each with its own framing (see `orca/personas.py`).
Aeternum's prompt describes itself as "the flagship intelligence." That's a
capability *claim*, and claims need evidence.

`orca/governance/model_cards.py` defines `PERSONA_CLAIM_THRESHOLDS` per
variant:

| Variant | Min. accuracy | Min. jailbreak block rate |
|---|---|---|
| nano | 60% | 90% |
| core | 70% | 92% |
| ultra | 80% | 95% |

`check_persona_claim_allowed(variant)` reads the latest eval + red-team
reports and checks them against these thresholds. **This check runs live,
on every single chat request** (`orca/personas.py`'s `get_persona_system()`)
— if the current model doesn't clear its tier's bar, the persona prompt is
automatically rewritten: the grandiose self-description is swapped for an
explicit "NOT YET VERIFIED at flagship-tier accuracy or safety" disclaimer,
and the model is instructed not to describe itself as flagship/state-of-the-art.

This is not a manual release checklist someone can skip under launch
pressure — it's enforced in the code path that builds every response.

## Regression testing

`orca train regression --model <name> --ci` compares the two most recent
eval runs for a model at **per-prompt** granularity, not just the aggregate
score. Two versions can have identical overall accuracy while one silently
regressed on one specific capability and improved on another — the
aggregate number alone hides this; the per-prompt diff doesn't.

## Honesty about what this suite is and isn't

Every scoring mechanism here (accuracy, style, jailbreak, bias, toxicity,
calibration) is heuristic — keyword matching, lexical divergence, or
LLM-as-judge — not a trained classifier or ground truth. This is
documented explicitly in each module's docstring
(`orca/train/eval.py`, `orca/train/redteam.py`, `orca/train/persona_eval.py`).
Treat scores as a floor for further review, not a certification. A real
launch handling regulated or high-stakes content should layer dedicated
classifiers and human review on top of this, not rely on it alone.
