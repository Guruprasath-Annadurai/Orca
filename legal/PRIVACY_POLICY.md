<!--
DRAFT — NOT LEGAL ADVICE. This document was generated to accurately describe
Orca's ACTUAL technical behavior as of the commit that added this file. It is
a starting point for a real privacy policy, not a substitute for review by a
licensed attorney in your jurisdiction before public launch. Data protection
law (GDPR, CCPA, etc.) has jurisdiction-specific requirements this draft does
not attempt to fully satisfy — have this reviewed before relying on it.
-->

# Orca Privacy Policy (Draft)

**Last updated:** see git history for this file.

## 1. What data Orca collects, and where it actually lives

This section describes what the software does, verified against the code,
not aspirational language.

| Data | What it is | Where it's stored | Retention |
|---|---|---|---|
| Account info | Email, display name, password hash (never plaintext) | SQLite (local) or Postgres (hosted deployments) | Until account deletion |
| Conversation history | Messages you send and Orca's responses | Local disk (JSON files) by default; Redis cache if the operator enables cross-instance session continuity | Configurable; see §4 |
| Uploaded documents | Files you upload for document Q&A | Local ChromaDB vector store, chunked and embedded | Until you delete the document or the session expires |
| Audit log | Security-relevant events (logins, chat requests, admin actions) | Hash-chained, tamper-evident ledger (SQLite or Postgres) | **Not user-deletable** — see §5 |
| Billing info | Stripe customer ID (not card numbers — those never touch Orca's servers, only Stripe's) | SQLite/Postgres | Until account deletion |
| 2FA secret | TOTP secret, if you enable two-factor authentication | SQLite/Postgres, not encrypted at rest beyond the database's own protections | Until you disable 2FA |

## 2. What Orca does NOT do

- Does not send your conversations to a third-party AI provider by default.
  Local deployments run entirely on the operator's own hardware via Ollama.
- Does not use your conversations to train models serving other users,
  unless the operator has explicitly opted into a training data pipeline
  and told you so separately.
- Does not sell personal data to third parties.
- Card numbers are never stored or processed by Orca directly — Stripe
  handles all payment card data under its own PCI-compliant infrastructure.

## 3. Cloud/third-party exceptions — when data DOES leave the deployment

Orca is local-first by default, but two features are explicit exceptions,
and both require the operator to opt in:

- **Distillation with a cloud teacher model**: if an operator uses the
  distillation pipeline with a Nvidia-hosted or other cloud model as a
  "teacher," the prompts sent to generate training data go to that
  provider's API. This is a training-data-generation step, not something
  that happens with live user conversations.
- **Stripe**: checkout and billing data is processed by Stripe under
  Stripe's own privacy policy (https://stripe.com/privacy). Orca only
  stores the Stripe customer ID, not payment details.

## 4. Data retention

- **Conversation history**: retained until you delete it or your session
  expires (default: 2 hours of inactivity for in-memory session state;
  persisted history remains on disk until explicitly deleted).
- **Uploaded documents**: retained until deleted via the document
  management interface, or until the session is cleared.
- **Account data**: retained until account deletion (see §5).

## 5. Your rights, and an important limitation on the audit log

You can request deletion of your account and associated conversation
history, documents, and billing linkage. As of this document's date, this
is a manual/administrative process — contact the operator directly. (A
self-service deletion endpoint is planned; check this document's git
history for updates once it ships.)

**Important limitation**: Orca maintains a hash-chained, tamper-evident
audit log for security and compliance purposes (who did what, when). This
log is intentionally NOT designed for individual entry deletion — removing
an entry would break the cryptographic chain that makes the log trustworthy
in the first place. Deleting a user's *account* does not retroactively
delete their *historical audit log entries*, though the log entries
themselves are pseudonymous references (a user ID), not a full account
record. If full historical erasure of even pseudonymous references is a
hard legal requirement in your jurisdiction (e.g., certain interpretations
of GDPR's right to erasure), consult counsel — this is a genuine design
tension between "tamper-evident security logging" and "individual erasure
rights," not something this draft resolves for you.

## 6. Security

- Passwords are hashed with PBKDF2-SHA256 (260,000 iterations), never
  stored in plaintext.
- Sessions use HMAC-signed tokens.
- Two-factor authentication (TOTP) is available and recommended.
- The audit log is cryptographically hash-chained — any tampering with
  historical records is detectable.
- See the project's model cards and red-team reports for AI-specific safety
  and bias testing results — published, not hidden.

## 7. Changes to this policy

This is a living document tied to the software's actual behavior. Material
changes will be reflected here; operators running their own Orca instance
are responsible for keeping this document in sync with their actual
configuration (e.g., if they enable cloud-based inference instead of local
Ollama, this document must be updated to reflect that).

## 8. Contact

Operators deploying Orca should replace this section with their own contact
information for privacy inquiries.
