#!/bin/bash
# Row 182: re-score GL and GLK20 on the owner's budget with the corrected Kind lines (merchant_db_match_real.py MATCH=words:
# "_v2" = Overture first then whole-word database matches; "_ov" = Overture only), privately. Needs BUDGET.
set -u
cd "$(dirname "$0")/../.."
P=decider_decider-4b_none_h100bf16st800
Q=_emp20_f0_ren50_dbep50_mislead_v1_alt10s_lk10_ov10_oth50_aux100_labrand255_laylabelled_shots_ev10soft_rs25
for kf in _ov _v2; do
  SIM=2 GROUPNAMES=1 KIND=rows KIND_FILE=$kf uv run python scripts/real_budget_eval.py build | tail -1
  for spec in "v2gkl:r182-gl" "v2gkl_mk40_20k6:r182-glk20"; do
    IFS=: read v tag <<<"$spec"
    echo "== $tag KIND_FILE=$kf"
    SIM=2 GROUPNAMES=1 KIND=rows KIND_FILE=$kf READER="adapter:${P}${Q}${v}_lora@${tag}" uv run --with modal python scripts/real_budget_eval.py modal 2>&1 | grep -v "^\s*$" | tail -1
  done
done
echo "== r182 corrected-lines scoring done"
