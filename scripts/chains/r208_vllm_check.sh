#!/bin/bash
# Row 208 E3 check: decider G4 (r190, seed 0) read by the HF loop and by vLLM (READER=vllm) on the local 3090, on 20 held-out v4
# households (1,200 items) and blind_v1; per-item agreement compared by scripts/vllm_compare.py.
set -eu
cd "$(dirname "$0")/../.."
UV='uv run --frozen --with transformers==5.17.0 --with flash-linear-attention --with peft>=0.21 --with torch==2.13.0 --with torchvision==0.28.0 python'
for S in realstyle_v4g_test blind_v1; do
  US=""; [ $S = realstyle_v4g_test ] && US="USERS=100000,100001,100002,100003,100004,100005,100006,100007,100008,100009,100010,100011,100012,100013,100014,100015,100016,100017,100018,100019"
  for R in hf vllm; do
    echo "== $S $R $(date -u +%H:%M:%S)"
    env FAMILY=decider MODEL=Mapika/decider-4b ADAPTER=decider_decider-4b_none_h100bf16st800_emp20_f0_ren50_dbep50_mislead_v1_alt10s_lk10_ov10_oth50_aux100_labrand255_laylabelled_shots_ev10soft_rs25v4g_lora CONDS=noctx LABELS=rand255 LAYOUT=labelled_shots ITEMS_SET=$S $US READER=$R TAG_SUFFIX=_chk$R $UV scripts/exp_decision_models.py 2>&1 | grep -E "top-1|vllm_slot|merged|Error|error" | tail -4
  done
done
echo "== r208 vllm check done $(date -u +%H:%M:%S)"
