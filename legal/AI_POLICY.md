<!--
DRAFT — operational AI policy and risk register, not a substitute for legal
review. Same honesty standard as legal/PRIVACY_POLICY.md and
legal/TERMS_OF_SERVICE.md: every mechanism referenced here is real, checked
against the code at the time this document was written, not aspirational.
Update this file whenever the underlying controls change — a policy
document that drifts from what the code actually does is worse than no
policy document at all.
-->

# Orca AI Policy & Risk Register (Draft)

**Last updated:** see git history for this file.

## 1. Purpose

This document states the policy that Orca's technical controls already
enforce, in one place, so it can be reviewed, audited, and held accountable
against — rather than the controls existing in code with no single
document describing what they're for.

## 2. Risk categories and current mitigations

| Risk | Mechanism | Where enforced |
|---|---|---|
| Model overclaims capability it hasn't demonstrated | Persona-claim gate — a variant's system prompt is automatically demoted from "flagship"/"chief scientist" framing to an explicit "not yet verified" disclaimer unless its latest eval clears variant-specific accuracy + jailbreak-block thresholds | `orca/governance/model_cards.py` (`PERSONA_CLAIM_THRESHOLDS`, `check_persona_claim_allowed`), enforced live in `orca/personas.py` on every request |
| Jailbreak / adversarial prompting | 10-probe red-team suite (roleplay bypass, DAN-style, prompt injection) with keyword refusal detection | `orca/train/redteam.py` — `run_jailbreak_suite()` |
| Differential treatment by demographic framing | 8 paired bias probes, lexical-divergence scoring | `orca/train/redteam.py` — `run_bias_probes()` — explicitly a triage signal for human review, not proof of bias |
| Toxic output | Static keyword-matched toxicity probes | `orca/train/redteam.py` — `run_toxicity_probes()` — explicitly sparse, not a moderation-classifier replacement |
| Confident-wrong answers on unknowable questions | Calibration probes — plausible-false-premise questions checking whether the model corrects the premise or builds on it | `orca/train/redteam.py` — `run_calibration_probes()` |
| Harmful/CSAM-adjacent/mass-casualty-weapon requests | Input moderation, hard block before generation | `orca/serve/moderation.py` — `BLOCK` category |
| Self-harm / suicide ideation | **Never blocked** — crisis resources injected into context instead, generation proceeds with care | `orca/serve/moderation.py` — `SUPPORT` category. This is a deliberate policy choice: refusing someone in crisis is not a safety practice this project follows. |
| Harassment / hate-speech-adjacent content | Logged for governance visibility, not blocked (avoids over-blocking legitimate discussion/critique) | `orca/serve/moderation.py` — `FLAG` category |
| Hallucinated sources / unverifiable factual claims | Mechanical citation check when RAG context exists (response must reference `[D#]` markers); system-prompt instruction to flag training-knowledge claims as unverified when no retrieval context exists | `orca/docs/citation_check.py`, `orca/personas.py`'s `_CITATION_DISCIPLINE_BLOCK` |
| PII exposure via uploaded documents | SSN/email/phone/credit-card redaction before chunking/embedding into the persistent vector store | `orca/docs/pii_redact.py` |
| Tampering with historical logs | Hash-chained audit log — any alteration to a historical entry is cryptographically detectable | `orca/audit.py` |
| Model quality regression between versions | Per-prompt eval diffing between the two most recent runs, CI-gateable | `orca/train/regression.py` |

## 3. Model release policy

No model variant's persona is allowed to represent itself at full capability
tier until it clears its tier's thresholds (see §2, persona-claim gate).
This is enforced automatically, at runtime, on every request — not a
manual checklist that can be skipped under launch pressure.

Recommended (not yet mandatory in tooling) release checklist before
deploying a new fine-tune to any variant:
1. `orca train eval --ollama <model> --ci` — accuracy/style gate
2. `orca train redteam --model <model> --ci` — safety gate
3. `orca train persona-eval --model <model> --variant <variant>` — persona-specific bar
4. `orca train regression --model <model> --ci` — no regression vs the previous version
5. `orca train card <variant>` — generate the signed model card documenting all of the above

## 4. Known limitations — stated plainly, not hidden

- Every red-team and moderation mechanism in §2 is heuristic (keyword/
  pattern/lexical-divergence based), not a trained classifier. Each is
  documented in its own module as "a floor, not a ceiling." A production
  deployment handling regulated or high-stakes content should layer a
  dedicated moderation/safety classifier on top of these, not rely on them
  alone.
- No third-party security audit or penetration test has been conducted on
  this system as of this document's drafting.
- Human evaluation (blind A/B against reference models) has a built
  harness (`orca/train/blind_ab.py`) but has not been executed — no human
  panel has rated Orca's outputs against anything yet.
- Bias probes cover 8 paired scenarios — a real, but narrow, sample of
  possible differential-treatment patterns. Absence of a flag is not
  evidence of absence of bias.

## 5. Risk register

| Risk | Likelihood | Impact | Current mitigation | Residual risk |
|---|---|---|---|---|
| Model gives confidently wrong answer on a factual question | High (calibration probes have measured this directly — see model cards) | Medium-High depending on use case | Citation discipline, calibration probes, persona claim gate demoting overclaiming | High — heuristic mitigations, not solved |
| Jailbreak produces harmful content | Medium (measured block rates vary by model — see red-team reports) | High | Jailbreak probe suite, input moderation BLOCK category | Medium — keyword-based, will miss novel attacks |
| PII leak via uploaded document | Low (redaction runs on every upload) | High if it occurs | Pattern-based PII redaction before embedding | Low-Medium — misses non-US ID formats, unstructured PII (names, addresses) |
| Audit log tampering goes undetected | Very Low | High | Hash chain + HMAC signature, verified via `verify_chain()` | Very Low — cryptographically enforced |
| Data breach of stored user data | Unknown (no pentest conducted) | High | Password hashing (PBKDF2), 2FA available, encrypted-in-transit (assumes TLS termination by the deployment) | **Unknown — no third-party security review has validated this** |
| Self-harm content mishandled | Low (explicit design decision documented) | High if mishandled | Never-block + crisis-resource injection policy | Low, but not clinically validated — this is not a substitute for a real crisis service |

## 6. Review cadence

This document should be re-reviewed whenever:
- A new model variant is released (tie to the release checklist in §3)
- Any module referenced in §2 changes its detection logic
- A red-team/moderation false negative is discovered in production use

## 7. Ownership

At this project's current stage (pre-team), the operator running any given
Orca deployment is the de facto owner of this policy. If Orca grows into an
organization with the divisions described in its own org-chart planning
(Trust & Safety, Security, Model Evaluation), ownership of this document
should move to a named executive function — the audit conducted alongside
this document recommended exactly that (a Chief Model Quality Officer role)
for the Model Evaluation function specifically.
