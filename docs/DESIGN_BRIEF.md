# Orca Design Brief — Screen by Screen

Written for use with a design generation tool (Claude/Figma/etc.). Grounded
in what Orca's code actually does as of this document's commit — every
screen listed maps to a real, working feature. Not aspirational, not
invented. One honest gap flagged up front: **the admin/governance surface
(audit log, model cards, red-team reports, metrics) currently has zero UI
— API-only.** That's the largest missing screen set for a genuine
enterprise claim, and it's called out explicitly below, not hidden.

## Logo & mark usage

Two marks provided:
1. **Geometric/tech orca** (black background, glitch/faceted white
   highlights, particle trail) — use as the primary hero/marketing mark,
   loading states, and splash screen. Too detailed for a 16-24px favicon.
2. **Minimal line-art orca** (single continuous stroke, white on
   transparent, scales cleanly from favicon to hero size) — use as the
   actual product UI mark (header logo, favicon, app icon). This is the
   one that needs to work at 24px in a header bar; the geometric mark
   doesn't scale down cleanly.

Do not use both marks interchangeably in the same context — geometric mark
= brand/marketing surface, line-art mark = product chrome.

## Design system foundation

- **Palette**: true black background (`#000000`/`#080808`), off-white text
  (`#e8e8e8`), pure white for emphasis (`#ffffff`), muted grays for
  secondary text (`#4a4a4a`–`#999999`) — matches the logo's black/white
  restraint, don't introduce a bright accent color that fights it. One
  exception: a desaturated signal color for state (green-ish for
  success/RAG-active, red-ish for errors/destructive actions) — keep these
  low-saturation so they read as "system state," not "brand color."
- **Typography**: monospace for data/technical chrome (session IDs, model
  names, timestamps, code) — reinforces the "engineering-grade tool" read
  the logo implies. Sans-serif for conversational content (chat messages,
  body copy) — monospace for an entire conversation reads as a terminal
  emulator, not an assistant.
- **Motion**: subtle, purposeful, never decorative. A "thinking" state
  should feel like the orca mark's glitch/particle texture activating
  briefly — tie animation language back to the brand mark, don't invent an
  unrelated spinner.
- **Grid & spacing**: generous whitespace on marketing/auth surfaces,
  denser information display on admin/governance surfaces (that audience
  wants data density, not marketing polish).

## Screen 1 — First-run / onboarding

Real gap: no onboarding flow exists today (flagged in the earlier audit).
Design for:
1. Welcome screen: line-art mark, one-line value prop ("Private
   intelligence. Your hardware. Your data."), single CTA to sign up.
2. Model selection: explain nano/core/ultra in plain language BEFORE the
   user picks — what each is good at, not just size. Include the honest
   framing already baked into the code: if a variant hasn't cleared its
   eval/safety thresholds, the UI should say so (the persona-claim gate
   already computes this — surface it visually, don't hide it behind a
   generic "beta" badge).
3. First message prompt with 3-4 example starters (already partially
   built — "Find investors," "Write pitch," etc. — keep this pattern, make
   it configurable per deployment).

## Screen 2 — Auth (sign in / sign up / forgot password / 2FA)

Already built functionally; elevate visually. Key states to design
explicitly:
- Sign in / sign up tab switch (exists)
- Forgot password → check-email confirmation (exists)
- **2FA challenge screen** (built in backend, NO dedicated UI yet — real
  gap): a clean 6-digit code input, clear messaging about which
  authenticator app step they're on, a "having trouble" link.
- **2FA setup screen** (also no UI yet): QR code render of the
  `provisioning_uri` the backend already returns, manual-entry secret as
  fallback text, a verify-code input before the backend will finalize
  enabling it.
- Email verification banner (exists) — keep, but make it dismissible state
  clearer.

## Screen 3 — Main chat (Orca Chat)

This is the core surface, already substantially built. Enterprise-level
elevation:
- **Sidebar**: session list with real titles (auto-titled from first
  message — exists), search, and a persistent model-variant indicator per
  session so switching context is never ambiguous.
- **Model pills** (nano/core/ultra) — already exist, keep, but the
  unavailable-model dimming (`.mpill-unavail`) needs a tooltip explaining
  WHY (not pulled in Ollama vs. doesn't clear the eval gate — these are
  different reasons, the UI currently can't distinguish them and should).
- **Message thread**: clear role separation (You / Orca), tool-use tags
  (exists — "web_search," "run_code" pills above a response), RAG source
  citations (exists as `[D1]`/`[D2]` markers — render these as clickable
  chips that open the source excerpt, not raw bracket text).
- **Explain button** (exists in backend, has frontend wiring) — the modal
  should read like a debug/trust panel: retrieval steps with timing,
  confidence bar, citations, contradiction warnings if any. This is a
  trust-building surface — design it to look authoritative, not like a
  dev console dump.
- **Citation compliance flag**: when a RAG answer used zero citations
  despite having document context (a real backend signal —
  `citation_compliance`), show a subtle "unverified against your documents"
  badge on that specific message. Don't bury this in a log only admins see.

## Screen 4 — Document Q&A (RAG)

- Attach button (paperclip, exists) → doc chips bar (exists) showing
  filename + chunk count + a remove control.
- Upload progress/processing state (extraction → PII redaction → chunking
  → embedding is a real multi-step pipeline — a determinate progress
  indicator with these actual stage labels builds more trust than a
  generic spinner, since the stages ARE real and meaningful).
- PII redaction disclosure: if redactions happened during upload (real
  backend data — `pii_redactions` count), tell the user plainly: "3 items
  redacted (2 emails, 1 phone number) before this document was stored."
  This is a trust-building disclosure, not something to hide.

## Screen 5 — Code Interpreter

- Run button on Python code blocks (exists) → inline output panel
  (stdout/stderr/exit code/duration — all real backend fields, exists).
- Sandbox boundary messaging: when a blocked import/builtin triggers the
  AST safety check, the error message should explain WHY in plain language
  ("`os` isn't available in the sandbox — this keeps code execution safe"),
  not just surface the raw `ImportError` string.

## Screen 6 — Voice input

- Mic button (exists), listening-state pulse animation (exists).
- Real gap: no visual transcript-in-progress feedback beyond the textarea
  filling in — consider a waveform or pulse tied to actual speech
  detection if the Web Speech API exposes it, for a more premium feel.

## Screen 7 — Vision (real gap, backend just shipped, zero UI)

New screen needed:
- Image attach button (separate from document attach — different pipeline)
- Preview thumbnail of the attached image before sending
- Clear model-capability messaging: if the active model isn't vision-capable
  (a real backend check — `is_vision_capable()`), tell the user BEFORE they
  waste effort attaching an image, not after a 400 error.

## Screen 8 — Knowledge Graph explorer (real gap, backend built, zero UI)

New screen needed — this is a genuine differentiator worth designing well:
- Entity list for the current session (real API: `GET /api/knowledge/{session_id}`)
- Click an entity → relationship graph view (nodes/edges, even a simple
  force-directed layout) showing one-hop neighbors (real API:
  `GET /api/knowledge/{session_id}/{entity_name}`)
- Honest framing badge: "extracted automatically from this conversation,
  not verified" — matches the backend's own honest scoping, don't let the
  UI imply more certainty than the data has.

## Screen 9 — Settings

- Profile (exists)
- **Billing/upgrade** (exists functionally via Stripe Checkout redirect) —
  needs a real pricing/plan comparison screen before the redirect, not just
  a single "Upgrade to Pro" button with no context on what changes.
- **2FA management** (setup/disable — backend exists, no UI, see Screen 2)
- **API keys** (backend exists — create/list/revoke — needs a real table UI)
- **Data & privacy**: right-to-delete (backend exists — real, tested) needs
  a genuine confirmation flow: show exactly what will be deleted (session
  count, document count) before requiring password re-entry, matching the
  backend's own detailed deletion report.

## Screen 10 — Admin / Governance dashboard (REAL GAP — biggest one)

This entire surface exists only as API endpoints right now. For an
"enterprise-level" claim this is not optional — it's the thing that makes
governance visible instead of theoretical:
- **Audit log viewer**: searchable/filterable table, a prominent "Verify
  chain integrity" button that calls the real `verify_chain()` endpoint and
  shows PASS/FAIL with the exact broken entry if tampering is detected.
- **Model cards**: one card per variant, rendering the real signed JSON —
  eval scores, safety scores, known limitations (auto-generated from real
  numbers), persona-claim-gate status front and center (approved/demoted,
  with the exact reason).
- **Red-team / eval history**: a real trend chart per model (regression
  testing already computes this diff — visualize it, don't make an admin
  read raw JSON).
- **Metrics dashboard**: request volume, error rate, latency percentiles
  per endpoint (real data from `/api/admin/metrics`) — this is what makes
  "production monitoring" feel real to a buyer doing due diligence.

## Screen 11 — Error / empty / loading states

Design these explicitly, don't leave them as an afterthought:
- Ollama unreachable ("model offline") — already surfaced in header, needs
  a fuller state when it blocks an entire action.
- Rate-limited (429, real backend behavior) — show the actual retry-after
  time from the response, not a generic "try again."
- Empty session (no messages yet) — exists, keep the example-prompt
  pattern.
- Empty knowledge graph / no documents uploaded — clear, non-alarming
  empty states.

## Accessibility (per the org chart's own Accessibility Department, taken seriously)

- Every icon-only button (attach, mic, send) needs a real `aria-label`, not
  just a `title` tooltip.
- Color contrast: verify the muted-gray text tiers against true black meet
  WCAG AA at minimum — the current palette is close but should be audited,
  not assumed.
- Keyboard navigation: full chat flow (compose, send, switch model,
  open Explain modal) should be operable without a mouse.
- Screen reader: streaming chat responses need an ARIA live region so
  assistive tech announces new content without re-reading the whole thread.

## Conversation design (tone, tied to real persona code)

- **Genesis** (nano): plain language, short sentences, explicit hedging
  language when uncertain — the UI could subtly reflect this with a
  slightly simpler visual density than Novus/Aeternum responses.
- **Novus** (core): visible reasoning structure in longer answers —
  consider rendering explicit "trade-off" language with light visual
  separation (not full step-numbering, just enough to signal structured
  thinking).
- **Aeternum** (ultra): when demoted by the persona-claim gate (a real,
  live possibility per the backend), the UI must not hide this — a visible
  "developing model, not yet verified at flagship tier" indicator on the
  model pill itself, not buried in a tooltip.

## What this document is not

A replacement for actual visual design work. This is the brief a designer
or design-generation tool needs — layouts, components, states, and the
honest data behind each screen. Producing final pixel-perfect visuals,
choosing exact typefaces, and validating against real users still needs
either a human designer or iterative work in the design tool itself.
