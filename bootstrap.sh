#!/usr/bin/env bash
# Fresh WSL/Ubuntu machine -> ready to run `just`. Installs only the two tools the justfile
# cannot install for itself (uv, just), then hands off to `just setup` and `just doctor`.
# Idempotent: anything already on PATH is skipped. Everything else (venv, DVC remote path,
# adapters) is the justfile's job; paper tools like pdftotext are the doctor's WARN, not ours.
set -euo pipefail
cd "$(dirname "$0")"

step() { printf '\n==> %s\n' "$*"; }

export PATH="$HOME/.local/bin:$PATH"   # where the uv installer puts uv, before it is in your shell rc

if command -v uv >/dev/null; then
    step "uv already installed: $(uv --version)"
else
    step "installing uv (curl -LsSf https://astral.sh/uv/install.sh | sh)"
    curl -LsSf https://astral.sh/uv/install.sh | sh
    command -v uv >/dev/null || { echo "uv not on PATH after install; open a new shell and re-run" >&2; exit 1; }
fi

if command -v just >/dev/null; then
    step "just already installed: $(just --version)"
else
    step "installing just (sudo apt install just)"
    sudo apt-get install -y just
fi

step "just setup   (uv sync, DVC remote path for this OS, dvc pull)"
just setup

step "just doctor"
just doctor
