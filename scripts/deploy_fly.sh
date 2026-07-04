#!/usr/bin/env bash
# Atheris — one-command Fly.io deploy
#
# First time:
#   1. Install flyctl:  curl -L https://fly.io/install.sh | sh
#   2. Login:           fly auth login
#   3. Run this script: ./scripts/deploy_fly.sh
#
# Subsequent deploys:
#   ./scripts/deploy_fly.sh

set -euo pipefail

APP="orca-demo"
REGION="ord"
VOLUME="orca_data"

# ── Check flyctl ─────────────────────────────────────────────────────────────
if ! command -v fly &>/dev/null; then
  echo "✗  flyctl not found. Install it:"
  echo "   curl -L https://fly.io/install.sh | sh"
  exit 1
fi

# ── Build wheel ──────────────────────────────────────────────────────────────
echo "▶  Building wheel..."
uv build --wheel
WHEEL=$(ls dist/orca_ai-*.whl | tail -1)
echo "   Built: $WHEEL"

# ── First-time setup ─────────────────────────────────────────────────────────
if ! fly apps list 2>/dev/null | grep -q "$APP"; then
  echo "▶  Creating app: $APP"
  fly apps create "$APP" --org personal

  echo "▶  Creating persistent volume (3GB)..."
  fly volumes create "$VOLUME" \
    --app "$APP" \
    --size 3 \
    --region "$REGION"

  echo "▶  Setting secrets..."
  # Generate a random JWT secret for the deployment
  JWT_SECRET=$(python3 -c "import secrets; print(secrets.token_hex(32))")
  fly secrets set ORCA_JWT_SECRET="$JWT_SECRET" --app "$APP"
fi

# ── Deploy ───────────────────────────────────────────────────────────────────
echo "▶  Deploying to Fly.io..."
fly deploy \
  --app "$APP" \
  --config fly.toml \
  --dockerfile Dockerfile.fly \
  --remote-only

echo ""
echo "✓  Deployed! Your Atheris instance is live at:"
echo "   https://${APP}.fly.dev"
echo ""
echo "   Admin: fly ssh console --app $APP"
echo "   Logs:  fly logs --app $APP"
echo "   Scale: fly scale count 1 --app $APP"
