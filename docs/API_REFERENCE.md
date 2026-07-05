# API Reference

Generated against the actual routes in `orca/serve/api.py` and
`orca/auth/routes.py` — every endpoint listed here exists in the code as of
this document's commit. If you find a discrepancy, the code is the source
of truth; file an issue.

Base URL: `http://localhost:7337` (default local install).

## Authentication

Most endpoints accept a Bearer token from `/api/auth/login` or
`/api/auth/signup`: `Authorization: Bearer <token>`. Endpoints marked
**optional auth** work without a token but behave differently when one is
provided (e.g. quota tracking, session ownership for right-to-delete).

### `POST /api/auth/signup`
Body: `{"email": str, "password": str, "name": str (optional)}`
Rate limited: 5/hour/IP. Returns `{"token", "user", "limits", "usage"}`.

### `POST /api/auth/login`
Body: `{"email": str, "password": str}`
Rate limited: 10/5min/IP. If the account has 2FA enabled, returns
`{"requires_2fa": true, "pending_token": str}` instead of a real token —
exchange it via `/api/auth/2fa/verify-login`.

### `POST /api/auth/2fa/setup` *(auth required)*
Generates a TOTP secret (not yet enabled). Returns `{"secret", "provisioning_uri"}` — the URI is scannable by any standard authenticator app.

### `POST /api/auth/2fa/enable` *(auth required)*
Body: `{"code": str}`. Finalizes 2FA — requires a valid code from the secret returned by `/setup`.

### `POST /api/auth/2fa/verify-login`
Body: `{"pending_token": str, "code": str}`. Exchanges the login's pending token + TOTP code for a real session token.

### `POST /api/auth/2fa/disable` *(auth required)*
Body: `{"code": str}`. Requires a valid current TOTP code, not just an active session.

### `POST /api/auth/account/delete` *(auth required)*
Body: `{"password": str}`. Full right-to-delete — cascades to every session's memory/documents/knowledge graph this account touched. See `orca/serve/account_delete.py` for exactly what is and isn't deleted (historical audit log entries are NOT deleted — tamper-evident by design).

Other auth endpoints: `GET /api/auth/me`, `POST /api/auth/forgot-password`,
`POST /api/auth/reset-password`, `POST /api/auth/change-password`,
`POST /api/auth/apikeys` (create), `GET /api/auth/apikeys` (list),
`DELETE /api/auth/apikeys/{key_id}`.

## Chat

### `POST /api/chat` *(optional auth)*
Body: `{"message": str, "session_id": str (optional), "model_variant": "nano"|"core"|"ultra" (optional)}`
Single-shot response. Rate limited (60/min/IP, floor under any per-user
tier quota). Runs input moderation before generation — see
`orca/serve/moderation.py`. Returns `{"response", "session_id", "used_tools", "plan"}`.

### `POST /api/stream` *(optional auth)*
Same body as `/api/chat`. Server-Sent Events. Event types: `session`,
`rag` (if documents are loaded for the session), `thinking`, `chunk`,
`done` (includes `message_id` for the Explain feature and
`citation_compliance`), `error`.

## Documents (RAG)

### `POST /api/docs/upload` *(optional auth)*
Multipart file upload + optional `session_id` form field. Extracts text,
redacts PII (SSN/email/phone/credit card — see `orca/docs/pii_redact.py`),
chunks, embeds into a per-session ChromaDB collection. Rate limited
20/min/IP.

### `GET /api/docs/list?session_id=...`
Lists documents uploaded to a session.

### `DELETE /api/docs/{doc_id}?session_id=...` *(optional auth)*
Removes a document and its chunks.

## Code execution

### `POST /api/code/run` *(optional auth)*
Body: `{"code": str, "language": "python", "session_id": str (optional)}`
Sandboxed execution — see `orca/code/sandbox.py` for the AST-level safety
checks (import whitelist, banned builtins). Rate limited 20/min/IP.

## Knowledge graph

### `GET /api/knowledge/{session_id}`
Lists entities extracted from this session's conversation.

### `GET /api/knowledge/{session_id}/{entity_name}`
Full relationship detail for one entity, plus one-hop neighbors in both
directions. See `orca/brain/knowledge_graph.py` for the honest scope
(per-session, LLM-extracted, not a production entity-resolution system).

## Explainability

### `GET /api/explain/{session_id}/{message_id}`
Full retrieval chain, query intelligence, citation DNA, sufficiency
confidence, and agent reasoning trace for a specific assistant message —
backs the "Explain this answer" UI feature.

## Billing

### `POST /api/billing/checkout` *(auth required)*
Query params: `tier` ("pro"|"enterprise"), `interval` ("month"|"year").
Creates a Stripe Checkout Session tied to the authenticated user via
`client_reference_id` — the mechanism that lets the webhook actually
upgrade the right account. Requires `STRIPE_SECRET_KEY` configured.

### `POST /webhook/stripe`
Stripe webhook receiver. Handles `checkout.session.completed` (upgrades
the account tier), `customer.subscription.deleted` and
`invoice.payment_failed` (downgrades back to free).

## Governance / admin *(admin auth required)*

- `GET /api/admin/audit` — recent audit log entries
- `GET /api/admin/audit/verify` — recomputes the hash chain, reports any tampering
- `GET /api/admin/audit/export` — signed export for compliance/legal handoff
- `GET /api/admin/governance/cards` — list all model cards
- `GET /api/admin/governance/cards/{variant}` — full model card detail
- `POST /api/admin/governance/cards/{variant}/generate` — regenerate from latest eval/red-team reports
- `GET /api/admin/metrics` — JSON metrics snapshot
- `GET /api/admin/stats` — user/tier counts

## Legal / policy (public, no auth)

- `GET /legal/terms`
- `GET /legal/privacy`
- `GET /legal/ai-policy`

## Monitoring (public, no auth — see note in SELF_HOSTING.md)

- `GET /metrics` — Prometheus exposition format

## Miscellaneous

- `GET /api/status` — model availability, uptime, session/training counts
- `GET /api/models` — which nano/core/ultra models are pulled in Ollama
- `GET /api/license` — current license tier
- `GET /api/sessions` / `POST /api/session/load` / `POST /api/session/save` — session management
- `PATCH /api/session/{session_id}/title` — rename a session
- `GET /api/session/{session_id}/export` — export as Markdown
- `POST /api/memory/recall` / `POST /api/remember` — long-term memory operations
- `POST /api/ultra` — multi-agent OrcaUltra pipeline (SSE)
