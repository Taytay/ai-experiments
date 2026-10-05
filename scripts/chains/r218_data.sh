#!/bin/bash
# Row 218 (a): training households 200-999 built once (data/interim/hh_cache) with their neighbour lists (data/interim/li_nb), so the
# 1,000-household arms only load them. One household at a time (low RAM, runs beside a training job).
set -eu
cd "$(dirname "$0")/../.."
export SHARED_WORLD=1 GROUPNAMES=1 REALSTYLE_V4=1 CTX=1
echo "start $(date -u +%H:%M:%S)"
uv run python - <<'PY' 2>&1 | grep --line-buffered -vE "Failed to load|warn|Loading|FutureWarning|Bytecode"
import sys, time
sys.path.insert(0, "scripts")
import li_decider as L
from two_tower import households
t, cache = time.time(), {}
for n, b in enumerate(households("train", range(200, 1000))):
    L.prepared(b, None, cache)
    if n % 50 == 0:
        cache.clear(); print(f"{n + 200} households ({time.time() - t:.0f}s)", flush=True)
print(f"done ({time.time() - t:.0f}s)")
PY
echo "== r218 data done $(date -u +%H:%M:%S)"
