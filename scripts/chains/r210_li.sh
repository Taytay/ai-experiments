#!/bin/bash
# Row 210: a late-interaction decision model (scripts/li_decider.py), one seed per arm (local GPU; on Modal the same arms run from scripts/modal_jobs/r210.json), then read on 50 held-out v4 households.
# a0 one document per category, MaxSim; a1 + the nearest earlier filings in the query (CTX); a2 soft interaction (UWE, tau_a 0.1); a3 + [CLS]
# cosine (SMART hybrid); a4 learned query-token weights; a5 per-filing candidates with MML (BELXTR); a6 MML + soft + hybrid.
set -eu
cd "$(dirname "$0")/../.."
export SHARED_WORLD=1 GROUPNAMES=1 REALSTYLE_V4=1
F='Failed to load|warn|Loading|FutureWarning|Bytecode|UNEXPECTED|position_ids|Notes|LOAD REPORT|^Key|^---|Writing model|deprecated'
echo "start $(date -u +%H:%M:%S)"
for a in "a0 MODE=doc" "a1 CTX=1" "a2 SOFT=0.1" "a3 HYBRID=1" "a4 QW=1" "a5 MODE=mml" "a6 MODE=mml SOFT=0.1 HYBRID=1"; do
  set -- $a
  arm=$1; shift
  [ -f models/encoders/li_r210_$arm/li_config.json ] && { echo "== $arm exists"; continue; }
  env ARM=$arm "$@" uv run python scripts/li_decider.py train 2>&1 | grep -vE "$F"
done
ARMS=a0,a1,a2,a3,a4,a5,a6 TEST_SEEDS=100000-100049 uv run python scripts/li_decider.py read 2>&1 | grep -vE "$F"
echo "== r210 done $(date -u +%H:%M:%S)"
