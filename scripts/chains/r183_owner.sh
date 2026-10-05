#!/bin/bash
# Row 183: the shared-world readers (G3, GC; seeds 0 and 1) on the owner's budget, which has no crowd line (no real crowd data yet):
# does training with "Others filed this payee as" lines hurt when the line is absent? Private, as real_budget_eval.py. Needs BUDGET.
set -u
cd "$(dirname "$0")/../.."
P=decider_decider-4b_none_h100bf16st800
Q=_emp20_f0_ren50_dbep50_mislead_v1_alt10s_lk10_ov10_oth50_aux100_labrand255_laylabelled_shots_ev10soft_rs25
for spec in ":v3g:r183-g-s0" "s1:v3g:r183-g-s1" ":v3gc:r183-gc-s0" "s1:v3gc:r183-gc-s1"; do
  IFS=: read s v tag <<<"$spec"
  echo "== $tag"
  SIM=2 GROUPNAMES=1 READER="adapter:${P}${s}${Q}${v}_lora@${tag}" uv run --with modal python scripts/real_budget_eval.py modal 2>&1 | grep -v "^\s*$" | tail -1
done
echo "== r183 owner scoring done"
