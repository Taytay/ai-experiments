#!/bin/bash
# Row 237: fcr with the busy-category correction on first-time payees (li_decider FREQ_TAU), a sweep on the rational households (real,
# obvious, seen-and-obvious names) and the owner's budget (decider v5 beside it); local GPU, whole budgets (no READ_EVERY).
set -u
cd "$(dirname "$0")/../.."
for tau in ${TAUS:-0 0.25 0.5 1 2}; do
  for p in "" obvious seenobv; do
    echo "== tau=$tau payees=${p:-real} $(date -u +%H:%M:%S)"
    FREQ_TAU=$tau RATIONAL_PAYEES=$p RS_V5=0 READ=rational_bank ARMS=li_r227_fcr SHARED_WORLD=1 GROUPNAMES=1 REALSTYLE_V4=1 OPTS=span \
      uv run python scripts/li_decider.py read 2>&1 | grep "^| li_"
  done
  echo "== tau=$tau owner"
  FREQ_TAU=$tau TOPK=1 DEC_TAGS=r231-dv5-s0,r231-dv5-s1 ENCS=li_r227_fcr scripts/chains/r218_owner.sh 2>&1 | grep -E "^\| LI "
done
echo "== r237 done $(date -u +%H:%M:%S)"
