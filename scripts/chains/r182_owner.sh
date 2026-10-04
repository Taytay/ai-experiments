#!/bin/bash
# Row 182: score the r182 adapters (and GK zero-shot with Kind lines) on the owner's budget, privately, one at a time.
# Needs BUDGET in the environment; the budget's items and scores stay in ~/.local/share/ynab-real-eval (real_budget_eval.py).
set -u
cd "$(dirname "$0")/../.."
P=decider_decider-4b_none_h100bf16
SIM=2 GROUPNAMES=1 KIND=rows uv run python scripts/real_budget_eval.py build | tail -1  # the Kind-line items (resumable scoring skips done arms)
Q=_emp20_f0_ren50_dbep50_mislead_v1_alt10s_lk10_ov10_oth50_aux100_labrand255_laylabelled_shots_ev10soft_rs25
for spec in "st800:v2gkl:r182-gl:rows" "st800:v2gkl_mk40_20k6:r182-glk20:rows" "st800:v2g_mk40_20k6:r182-gk20:" \
            "st800s1:v2g:r182-g-s1:" "st800s1:v2g_mk40_5k:r182-gk-s1:" "st800:v2g_mk40_5k:r181-gk-s0:rows"; do
  IFS=: read st v tag kind <<<"$spec"
  echo "== $tag KIND=$kind"
  SIM=2 GROUPNAMES=1 KIND=$kind READER="adapter:${P}${st}${Q}${v}_lora@${tag}" uv run --with modal python scripts/real_budget_eval.py modal 2>&1 | grep -v "^\s*$" | tail -2
done
echo "== r182 owner scoring done"
