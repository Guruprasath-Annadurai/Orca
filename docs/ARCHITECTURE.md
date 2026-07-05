# Architecture Overview

Orca is organized in layers, roughly following the idea that "the LLM is
one subsystem, not the product." This document maps that framing to the
actual code — every module named here exists; this is not aspirational.

## Model layer
Local inference via Ollama (`orca/brain/providers.py`). Three variants —
nano/core/ultra — each with a distinct persona system prompt
(`orca/personas.py`) rather than one generic assistant identity.

## Agent / reasoning layer
`orca/brain/agent.py`'s `AgentLoop` — plan → act (tools) → respond →
reflect. Not a single next-token prediction call: the planner decides if
tools are needed, tool results get incorporated, and complex responses go
through a self-critique/improve pass before returning.

## Memory layer
`orca/brain/memory.py` — four sub-systems: short-term (in-context, now
budget-compressed — see Context Intelligence below), long-term (ChromaDB
vector recall across sessions), episodic (structured session logs), and
semantic (distilled facts/concepts).

## Context Intelligence
`orca/brain/context_intelligence.py` — replaces blunt turn-count
truncation with budget-aware compression: once conversation history
exceeds a character budget, older turns get summarized via one LLM call
rather than silently dropped, while recent turns stay verbatim. Fixed a
real bug found during development: the agent's history buffer had no size
cap at all before this existed.

## Knowledge layer
- **RAG** (`orca/docs/`): document upload → PII redaction → chunking →
  embedding → a 7-stage retrieval pipeline (query rewrite, multi-signal
  recall, RRF fusion, reranking, sufficiency check, citation tagging) —
  see `orca/docs/pipeline.py`.
- **Knowledge graph** (`orca/brain/knowledge_graph.py`): per-session,
  LLM-extracted entity/relationship triples. Explicitly not a production
  graph database — no cross-session entity resolution, no fact versioning.

## Multi-agent layer
`orca/variants/ultra.py`'s `OrcaUltra` — sub-agent pods (researcher, coder,
analyst, writer, critic, architect), each with its own system prompt,
coordinated by an executive layer that merges their outputs.

## Tool layer
`orca/tools/` — web search, code execution, file operations, shell,
memory recall. The agent decides when to use them, reports what it used
(`used_tools` in API responses), and never fabricates a tool call it
didn't actually make.

## Governance layer
- **Audit log** (`orca/audit.py`) — hash-chained, tamper-evident. Verified
  under real concurrent multi-process writes via a Postgres advisory lock.
- **Model cards** (`orca/governance/model_cards.py`) — signed, derived
  from actual eval/red-team data, gate persona claims at runtime.
- **Eval suite** (`orca/train/eval.py`, `persona_eval.py`, `redteam.py`,
  `regression.py`) — accuracy, persona-specific benchmarks, safety probes
  (jailbreak/bias/toxicity/calibration), and per-prompt regression testing.
- **Input moderation** (`orca/serve/moderation.py`) — three-tier action
  (block/support/flag), not binary. Self-harm content is never blocked —
  crisis resources get injected instead.

## Platform layer
- **Auth**: PBKDF2 password hashing, HMAC session tokens, RBAC, API keys,
  TOTP-based 2FA (`orca/auth/`).
- **Database**: dual-backend — SQLite by default (zero setup), Postgres
  opt-in via `ORCA_DATABASE_URL` for multi-instance deployments
  (`orca/auth/db.py`).
- **Sessions**: in-memory by default, Redis opt-in via `ORCA_REDIS_URL`
  for cross-instance continuity (`orca/serve/session_store.py`).
- **Rate limiting**: IP-based, dual-backend same as sessions
  (`orca/serve/ratelimit.py`).
- **Monitoring**: `/metrics` (Prometheus) and `/api/admin/metrics` (JSON)
  — in-memory, single-instance (`orca/serve/metrics.py`).
- **Backup/DR**: `orca/ops/backup.py` — SQLite online backup API or
  `pg_dump`, externally scheduled.

## Honest gaps (as of this document)

- No custom training infrastructure, distributed systems, or inference
  engine — riding entirely on Ollama + standard fine-tuning tooling.
- No live Kubernetes deployment, no production monitoring dashboard
  (Prometheus format is exposed; nothing scrapes it yet by default).
- No vision/multimodal input pipeline.
- No third-party security audit or penetration test.
- Human evaluation harness exists (`orca/train/blind_ab.py`) but has never
  been run against a real reference model or real human raters.
