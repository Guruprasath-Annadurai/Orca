# Claude Design Prompts — Screen by Screen

Copy-paste prompts for a design generation tool. Run the **Base Style
Prompt** first to establish the visual language, then run each screen
prompt in the same session/context so they stay consistent with each
other. Every data field referenced (chunk counts, eval scores, etc.) is
real — pulled from the actual API, not placeholder text to invent.

---

## Base Style Prompt (run first)

```
Design system for "Orca" — a private, local-first AI platform. Enterprise-
grade, engineering-tool aesthetic, not a consumer chat app.

Palette: true black background (#000000/#080808), off-white primary text
(#e8e8e8), pure white for emphasis (#ffffff), muted gray tiers for
secondary/tertiary text (#4a4a4a to #999999). One desaturated signal green
for success/active states, one desaturated signal red for errors/
destructive actions — both low-saturation, never bright/neon. No other
accent colors.

Typography: monospace font family (JetBrains Mono or similar) for all
technical/data chrome — session IDs, model names, timestamps, metrics,
code. Clean sans-serif (Inter or similar) for conversational content and
body copy. Never use monospace for full paragraphs of assistant responses
— it reads as a terminal, not an assistant.

Logo: a minimal single-continuous-stroke orca line-art mark, white on
transparent, used in headers/favicon at small scale. A separate more
detailed geometric/faceted orca mark with a glitch/particle trail is
reserved for marketing/splash contexts only — do not use it in product
chrome.

Motion: subtle, purposeful. Loading/thinking states use a soft pulse or
particle-glitch texture that echoes the geometric logo mark, not a generic
spinner.

Spacing: generous whitespace on auth/marketing screens, denser information
layout on admin/data screens — the admin audience wants density, not
polish-for-its-own-sake.

Overall feel: a precision instrument. Confident, quiet, technical — the
opposite of a bubbly consumer AI assistant.
```

---

## Screen 1 — First-run onboarding

```
Design a 3-step onboarding flow for Orca, black background, white
line-art orca mark centered on step 1.

Step 1 — Welcome: centered logo mark, one-line value proposition text
"Private intelligence. Your hardware. Your data." below it, single
primary white button "Get Started."

Step 2 — Model selection: three cards side by side, labeled Genesis
(nano), Novus (core), Aeternum (ultra). Each card shows a one-sentence
description of what that tier is actually good at (Genesis = everyday
simple/honest assistant, Novus = deep reasoning partner, Aeternum =
flagship cross-domain synthesis). If a tier hasn't cleared its internal
quality/safety threshold, show a small amber "not yet verified" badge on
that card instead of hiding the limitation — this is a real, live signal
from the backend, must be visually present, not decorative.

Step 3 — First message: a single centered input with placeholder "Ask
Orca anything," and 4 example-prompt chips below it in a 2x2 grid.
```

---

## Screen 2a — Auth: sign in / sign up

```
Design a centered auth card, max-width ~420px, black background, bracket-
corner decorative frame (thin white corner brackets, not a full border).

Header: line-art orca mark, wordmark "ORCA" below it, tagline "Private
Intelligence / Powered by Atheris" in muted gray monospace.

Tab switcher: two pill-shaped tabs "Sign In" / "Sign Up," active tab
white background with black text, inactive tab transparent with muted
text.

Sign In form: email input, password input (both dark fields with subtle
border, light placeholder text), primary white full-width button "Sign
In." Below: "No account? Sign up free" (link) and "Forgot password?"
(smaller, muted link).

Footer: three small links, muted gray, centered — "Terms of Service ·
Privacy Policy · AI Policy."
```

---

## Screen 2b — Auth: 2FA setup

```
Design a 2FA setup screen, same card/frame style as the auth screen.

Header: "Set Up Two-Factor Authentication," one sentence of context below.

Center: a QR code (generated from a real otpauth:// URI), with a "Can't
scan? Enter this code manually" toggle revealing a monospace secret string
with a copy button.

Below: a single 6-digit code input (large, spaced digit boxes, monospace),
label "Enter the code from your authenticator app," primary button
"Verify & Enable."

Error state: red-tinted inline message "Invalid code — check your app and
try again" below the input.
```

---

## Screen 2c — Auth: 2FA challenge (login gate)

```
Design a 2FA challenge screen shown mid-login, same card style.

Header: "Two-Factor Verification," subtext "Enter the 6-digit code from
your authenticator app."

Center: 6-digit code input (large, spaced boxes, monospace, auto-focus).
Primary button "Verify."

Below: muted small text "Trouble accessing your authenticator? Contact
support" as a fallback link — no fake "resend code" option since TOTP
codes aren't sent, don't imply otherwise.
```

---

## Screen 3 — Main chat interface

```
Design the primary chat screen for Orca — three-column-feeling layout
but really a sidebar + main pane.

Left sidebar (dark, slightly lighter than main background, ~260px):
session list, each row showing an auto-generated session title and a
small model-variant tag (nano/core/ultra, tiny colored dot). Search input
at top. "New Session" button.

Top header bar: line-art orca logo left, connection status (green dot
"Online" / red dot "Offline") center-left, three model-variant pills
(NANO/CORE/ULTRA) center, user avatar + menu right.

Main chat area: message thread, clear visual separation between "You"
(right-aligned or left-aligned, your choice, but distinct) and "Orca"
messages. Assistant messages in sans-serif, generous line height. Above
an assistant message that used tools, show small pill tags: "web_search,"
"run_code" etc. Below a message that cited documents, show numbered
citation chips [D1] [D2] that look clickable, not raw bracket text.

Below the final assistant message in a thread, show a small ghost/outline
"Explain" button with a small magnifying-glass icon.

Bottom input area: multi-line textarea, attach-document icon button,
attach-image icon button, mic icon button, send button (arrow icon).
Below the input: quick command hints ("/web," "/run," "/remember") and
the three model pills repeated for quick switching.
```

---

## Screen 4 — Document upload & citations

```
Design the document attachment experience within the chat input area.

Doc chips bar: horizontal row of small pill chips above the input, each
showing a document icon, filename (truncated), chunk count, and an × to
remove.

Upload progress state: when a file is uploading, show a determinate
progress indicator with REAL stage labels cycling through: "Extracting
text..." → "Redacting sensitive data..." → "Chunking..." → "Embedding..."
— these are actual pipeline stages, not decorative filler text.

Post-upload confirmation message (rendered as a system-style note in the
chat thread, not a toast): "resume.pdf uploaded — 12 chunks indexed. 3
items redacted (2 emails, 1 phone number) before storage." — the
redaction count must be shown plainly, this is a real trust disclosure.

Citation detail: clicking a [D1] chip opens a small popover showing the
source filename, chunk number, and a text excerpt.
```

---

## Screen 5 — Code interpreter

```
Design a code block with an execution affordance, embedded in a chat
message.

Code block: dark background slightly different shade from the page
background, syntax-highlighted Python, a thin header bar above the code
showing the language label and a "▶ Run" button (outline style, right-
aligned in the header bar).

Output panel: appears directly below the code block after running,
visually connected (no gap), showing stdout in default text color, stderr
in the muted red signal color, and a small monospace footer line showing
exit code and duration in ms (e.g. "exit 0 · 340ms").

Blocked-execution state: if the sandbox rejects the code (disallowed
import), show the output panel with a clear plain-language explanation —
"`os` isn't available in the sandbox — this keeps code execution safe" —
not a raw Python traceback.
```

---

## Screen 6 — Voice input

```
Design the voice input active state for the chat input bar.

Mic button: when listening, the button border and icon shift to the
muted red signal color with a soft pulsing glow animation (steady 1s
pulse, not frantic).

Live transcript: as speech is recognized, text fills the input textarea
in real time with a very subtle fade-in per word, distinguishing
interim (lighter/italic) vs final (solid) recognized text.
```

---

## Screen 7 — Vision / image input

```
Design an image attachment flow, parallel to but visually distinct from
document attachment (different icon — an image/photo icon vs a document
icon).

Attach state: clicking the image icon opens a file picker; once selected,
show a small thumbnail preview chip (not just a filename) above the input,
with an × to remove.

Capability-check message: if the active model isn't vision-capable, show
this BEFORE the user sends, as an inline warning directly under the
thumbnail: "The current model doesn't support images. Switch to a vision-
capable model or continue with text only." — must appear pre-send, not as
a failure after the fact.
```

---

## Screen 8 — Knowledge graph explorer

```
Design a knowledge graph panel, accessible from a small "Knowledge" tab or
icon in the session view.

Entity list: left column, simple list of entity names extracted from the
current session, small type badge per entity (person/organization/
technology/place/concept), mention count shown as a small number.

Graph view: right/main area, clicking an entity shows a simple node-and-
edge diagram — the selected entity as a center node, one-hop related
entities as surrounding nodes, edge labels showing the relationship
predicate (e.g. "founded," "develops"). Keep this understated — thin white
lines on black, small text labels, not a colorful force-directed mess.

Honesty badge: small persistent label at the top of this panel: "Extracted
automatically from this conversation — not independently verified." This
must always be visible when the panel is open, not just on first use.
```

---

## Screen 9 — Settings

```
Design a settings screen with a left-hand tab list: Profile, Billing,
Security, API Keys, Data & Privacy.

Billing tab: show current plan name and price, a simple 2-3 column plan
comparison table (Free / Pro / Enterprise) with real feature differences
(message limits, model access, priority), single "Upgrade" button per
paid tier that leads to checkout.

Security tab: 2FA status (enabled/disabled with a toggle-like button
leading to the setup flow from Screen 2b), password change form.

API Keys tab: a simple table — key name, created date, last used date,
a masked key preview, revoke button per row. "Create New Key" button
at top, which on click shows the full raw key ONCE in a copyable code
block with a clear "this won't be shown again" warning.

Data & Privacy tab: a "Delete My Account" section, red-outlined
(destructive but not screaming), which on click opens a confirmation
modal showing REAL specifics before proceeding: "This will delete 3
sessions, 7 documents, and your account. Historical audit log entries are
retained per our data policy. Enter your password to confirm." Password
input, final red "Delete Account" button.
```

---

## Screen 10 — Admin / Governance dashboard (the real gap)

```
Design an admin dashboard, denser and more data-forward than the rest of
the product — this audience wants information density, not marketing
polish. Left nav: Audit Log, Model Cards, Eval History, Metrics.

Audit Log view: a searchable/filterable table (columns: timestamp, event,
user, detail preview). A prominent button at the top "Verify Chain
Integrity" — clicking it runs a real check and shows a large PASS (green)
or FAIL (red) result; on FAIL, show the exact sequence number and reason
of the broken entry, not a vague "tampering detected."

Model Cards view: one card per variant (Genesis/Novus/Aeternum) as a
detail panel — base model, version, signature (truncated hash, monospace),
a big "Persona Claim: Approved / Not Yet Verified" status badge with the
exact reason text beneath it, eval scores (accuracy %, style /10, speed
tok/s) as simple stat blocks, safety scores (jailbreak block rate, bias
flag rate, calibration score) as a second row of stat blocks, and a
"Known Limitations" list rendered exactly as generated by the backend —
do not paraphrase or soften this text.

Eval History view: a simple line chart per model showing overall score
over time, with regression points marked distinctly (a small red marker
where regression_count > 0 between two runs).

Metrics view: per-endpoint table — request count, error rate, p50/p95/p99
latency — plus an uptime counter and overall error rate at the top as
large stat numbers.
```

---

## Screen 11 — Error / empty / loading states

```
Design a small set of system states, consistent black/white/muted style.

Offline state: header status dot turns red, a banner appears above the
chat input: "Ollama is offline — start it locally to continue," with a
muted retry button.

Rate-limited state: an inline error message in the chat thread showing
the exact retry time from the real API response: "Rate limit reached — try
again in 47 seconds," not a generic "try again later."

Empty session: centered, muted, the orca line-art mark at low opacity,
text "Start a conversation" with the example-prompt chips from Screen 1
reused here.

Empty knowledge graph: centered text "No entities extracted yet — they'll
appear here as you chat," no error styling, this is a normal/expected
state not a failure.
```
