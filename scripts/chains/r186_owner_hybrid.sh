#!/bin/bash
# Row 186: the owner's budget with the encoder's alias matches (ALIAS_EMB=payee_enc_v1, cosine >= 0.95) added to the n-gram aliases, the
# similar-payee rows unchanged; read by G seeds 0 and 1, paired against their SIM=2 reads. Needs BUDGET.
set -u
cd "$(dirname "$0")/../.."
export SIM=2 GROUPNAMES=1 ALIAS_EMB=payee_enc_v1 ALIAS_EMB_MIN=0.95
uv run python scripts/real_budget_eval.py build | tail -1
P=decider_decider-4b_none_h100bf16st800
Q=_emp20_f0_ren50_dbep50_mislead_v1_alt10s_lk10_ov10_oth50_aux100_labrand255_laylabelled_shots_ev10soft_rs25v2g_lora
for spec in ":r181-g-s0" "s1:r182-g-s1"; do
  IFS=: read s tag <<<"$spec"
  echo "== $tag"
  READER="adapter:${P}${s}${Q}@${tag}" uv run --with modal python scripts/real_budget_eval.py modal 2>&1 | grep -v "^\s*$" | tail -1
done
echo "== r186 hybrid scoring done"
