#!/bin/bash
# Row 234: the rational households with obvious novel payees and / or category synonyms (scripts/rational_budgets.py RATIONAL_PAYEES,
# RATIONAL_CATS), same purchases as rational_bank; ARMS read on each of the four versions, local GPU.
set -u
cd "$(dirname "$0")/../.."
ARMS=${ARMS:-li_r227_fcr,li_r233_v6b,li_r233_v6p}
F='Failed to load|warn|Loading|it/s\]|FutureWarning|Bytecode|^\s*$'
for v in ${VARIANTS:-: obvious: :synonyms obvious:synonyms}; do  # payees:categories (categories: synonyms | loose)
  IFS=: read p c <<<"$v"
  echo "== payees=${p:-pool} categories=${c:-plain} $(date -u +%H:%M:%S)"
  RATIONAL_PAYEES=$p RATIONAL_CATS=$c RS_V5=0 READ=rational_bank ARMS=$ARMS SHARED_WORLD=1 GROUPNAMES=1 REALSTYLE_V4=1 OPTS=span \
    uv run python scripts/li_decider.py read 2>&1 | grep -vE "$F" | grep -E "^\| li_|^\| model"
done
echo "== r234 done $(date -u +%H:%M:%S)"
