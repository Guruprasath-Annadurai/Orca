# Self-Hosting Orca

Orca is local-first by default — no external API calls for chat unless you
explicitly opt into a cloud teacher model for distillation. This guide
covers running it yourself, from a laptop to a real deployment.

## Quick start (local, single-user)

```bash
git clone https://github.com/Guruprasath-Annadurai/Orca.git
cd Orca
uv pip install -e .

# Install Ollama and pull a model
curl -fsSL https://ollama.com/install.sh | sh
ollama pull llama3.1:8b   # or your preferred model

orca serve
```

Opens on `http://localhost:7337`. Data lives at `~/.orca` by default (SQLite,
ChromaDB, memory files, audit log). No cloud dependency, no account with a
third party required to run this.

## Configuration

Copy `.env.example` to `.env` and fill in only what you need — most fields
are optional:

```bash
cp .env.example .env
```

Key variables:
- `ORCA_LICENSE_SECRET` — required before generating/verifying license keys at scale
- `STRIPE_SECRET_KEY` / `STRIPE_WEBHOOK_SECRET` — only if you're taking payments
- `ORCA_DATABASE_URL` — set this to switch from SQLite to Postgres (needed for multi-instance deployments — see below)
- `ORCA_REDIS_URL` — set this to enable cross-instance session continuity (also multi-instance only)
- `NVIDIA_API_KEY` — only if using a cloud teacher model for distillation (`orca/train/distill.py`), never used for live chat

**Do not set `ORCA_HOME` to a value with an unexpanded `~`** — see the
comment in `.env.example` for why (a real bug this project found and fixed
in its own testing: `~` doesn't auto-expand in every code path that reads
environment variables, and a raw `~/.orca` string can silently redirect all
your data to the wrong location).

## Scaling beyond one instance

Single-instance/local deployments work fine on SQLite with in-memory
sessions — this is the default and requires no extra setup.

Once you're running multiple API instances behind a load balancer, two
things need to move from "in-process" to "shared":

1. **Database**: set `ORCA_DATABASE_URL=postgresql://...` — see
   `docker-compose.yml` for a ready-to-use Postgres service definition.
   Migrate existing SQLite data with `orca/auth/migrate_to_postgres.py`.
2. **Sessions**: set `ORCA_REDIS_URL=redis://...` — without this, a
   user's conversation continuity breaks if their next request lands on a
   different instance than the one that handled their last message.

## Docker / docker-compose

```bash
docker compose up -d
```

Brings up Ollama, the Orca API, and (if uncommented in `docker-compose.yml`)
Postgres and Redis. See the comments in that file — Postgres/Redis services
are opt-in, commented out by default, since most deployments don't need
them.

## Fly.io deployment

```bash
./scripts/deploy_fly.sh
```

See `fly.toml` and `Dockerfile.fly`. This deploys a single-instance
configuration to Fly.io's platform. Note: as of this writing, this
deployment path has been built and configured but not verified against a
live Fly.io environment — treat it as a documented starting point, test it
yourself before depending on it for a real launch.

## Backups

Zero backup automation exists unless you set it up. Run manually or via
cron:

```bash
orca backup --keep 14        # creates a backup, prunes to the last 14
orca backups                 # list existing backups
orca restore <path> --yes    # restore from a backup (SQLite only)
```

Recommended crontab entry (daily at 3am):
```
0 3 * * * cd /path/to/orca && uv run orca backup --keep 14
```

See `orca/ops/backup.py` for exactly what this does and doesn't cover
(Postgres restore uses `pg_restore` directly, not automated here).

## Monitoring

- `GET /metrics` — Prometheus exposition format, no auth (standard scrape
  convention — put a firewall/reverse-proxy rule in front of this port if
  it's internet-reachable)
- `GET /api/admin/metrics` — JSON snapshot, requires admin auth

Both are in-memory, single-instance metrics — see `orca/serve/metrics.py`
docstring for the honest scope of what this does and doesn't cover for
multi-instance deployments.

## Model training / fine-tuning

Orca ships with a fine-tuning pipeline for three variants (nano/core/ultra),
but training a real checkpoint requires a real GPU (an H100-class card is
what this project's own testing targeted — see `orca/train/variants.py`
for exact VRAM requirements per variant). See `orca/train/` for the full
pipeline: data seeding, fine-tuning, evaluation, red-teaming, and
distillation from a teacher model.
