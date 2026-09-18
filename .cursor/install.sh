#!/usr/bin/env bash
# Idempotent Cloud Agent bootstrap for the lead-enrichment agent.
# Installs the package (with dev extras) plus the Playwright Chromium
# browser and its OS libraries. Safe to run repeatedly.
set -euo pipefail

cd "$(dirname "$0")/.."

# Python build/runtime prerequisites.
sudo apt-get update -qq
sudo apt-get install -y --no-install-recommends python3-pip

# Install the project system-wide so `python3 -m lead_enrichment` and
# `pytest` work without activating a virtualenv. Installing outside the
# repo checkout keeps dependencies intact when the workspace is refreshed.
sudo python3 -m pip install --break-system-packages -e ".[dev]"

# Chromium OS libraries (apt, needs root) then the browser binary itself
# (downloaded into the invoking user's ~/.cache/ms-playwright). Both steps
# no-op when already satisfied.
sudo "$(command -v playwright)" install-deps chromium
playwright install chromium

echo "Cloud Agent environment ready."
