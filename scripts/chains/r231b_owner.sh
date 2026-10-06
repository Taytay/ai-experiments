#!/bin/bash
# Row 231 (REAL-6 ablation): decider G4 recipe on 100% v5 real-style episodes (no REAL-6), read on the owner's budget as r231_owner.sh.
set -u
cd "$(dirname "$0")/../.."
export BUDGET=${BUDGET:-96c06c41-f26f-4a44-98ed-eaba2f471e1e} SIM=2 GROUPNAMES=1
A=decider_decider-4b_none_h100bf16st800_emp20_f0_ren50_dbep50_mislead_v1_alt10s_lk10_ov10_oth50_aux100_labrand255_laylabelled_shots_ev10soft_rs100v5g_lora
READER="adapter:${A}@r231-dv5only-s0" uv run --with modal python scripts/real_budget_eval.py modal 2>&1 | grep -v "^\s*$" | tail -1
PYTHONPATH=scripts uv run python - <<'PY'
import json, numpy as np, real_budget_eval as RB
items = {it["id"]: it for it in json.loads((RB.OUT / "items_grp_sim2.json").read_text())["items"] if it["answer"] >= 0}
rs = [r for r in map(json.loads, open(RB.OUT / "scores_r231-dv5only-s0_split_grp_sim2.jsonl")) if r["id"] in items]
print(f"r231-dv5only-s0: {100 * np.mean([int(np.argmax(r['lp'])) == items[r['id']]['answer'] for r in rs]):.1f}% right first (n={len(rs)})")
PY
echo "== r231b owner done $(date -u +%H:%M:%S)"
