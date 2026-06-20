#!/usr/bin/env bash
# Atheris PyPI publish script
# Usage:
#   ./scripts/publish.sh          # build + upload to PyPI
#   ./scripts/publish.sh --test   # upload to TestPyPI first
#
# Prerequisites:
#   pip install build twine
#   Set PYPI_TOKEN env var or use ~/.pypirc

set -euo pipefail

TEST_PYPI=false
[[ "${1:-}" == "--test" ]] && TEST_PYPI=true

echo "▶  Cleaning previous builds..."
rm -rf dist/ build/ *.egg-info

echo "▶  Building wheel + sdist..."
python -m build

echo "▶  Built packages:"
ls -lh dist/

if $TEST_PYPI; then
  echo "▶  Uploading to TestPyPI..."
  python -m twine upload \
    --repository-url https://test.pypi.org/legacy/ \
    --username __token__ \
    --password "${PYPI_TOKEN:?PYPI_TOKEN not set}" \
    dist/*
  echo "✓  Uploaded. Install with:"
  echo "   pip install --index-url https://test.pypi.org/simple/ atheris-ai"
else
  echo "▶  Uploading to PyPI..."
  python -m twine upload \
    --username __token__ \
    --password "${PYPI_TOKEN:?PYPI_TOKEN not set}" \
    dist/*
  echo "✓  Published! Install with:"
  echo "   pip install atheris-ai"
  echo "   pip install atheris-ai[docs]   # PDF/DOCX support + ChromaDB"
  echo "   pip install atheris-ai[all]    # everything"
fi
