#!/bin/bash
# Row 231: decider G4 trained with v5 real-style episodes, read on the owner's budget exactly as r190's G4 (SIM=2 GROUPNAMES=1, decider's
# items, Modal private scoring: nothing kept there), then % right first for v4 (r190-g) and v5 (r231-dv5) seeds side by side (aggregates).
set -u
cd "$(dirname "$0")/../.."
export BUDGET=${BUDGET:-96c06c41-f26f-4a44-98ed-eaba2f471e1e} SIM=2 GROUPNAMES=1
P=decider_decider-4b_none_h100bf16st800
Q=_emp20_f0_ren50_dbep50_mislead_v1_alt10s_lk10_ov10_oth50_aux100_labrand255_laylabelled_shots_ev10soft_rs25v5g_lora
for spec in ":r231-dv5-s0" "s1:r231-dv5-s1"; do
  IFS=: read s tag <<<"$spec"
  echo "== $tag $(date -u +%H:%M:%S)"
  READER="adapter:${P}${s}${Q}@${tag}" uv run --with modal python scripts/real_budget_eval.py modal 2>&1 | grep -v "^\s*$" | tail -1
done
PYTHONPATH=scripts uv run python - <<'PY'
import json, numpy as np, real_budget_eval as RB
items = {it["id"]: it for it in json.loads((RB.OUT / "items_grp_sim2.json").read_text())["items"] if it["answer"] >= 0}
first = {it["id"] for it in items.values() if it.get("payee_seen") is False or it.get("seen") is False}
for t in ("r190-g-s0", "r190-g-s1", "r231-dv5-s0", "r231-dv5-s1"):
    f = RB.OUT / f"scores_{t}_split_grp_sim2.jsonl"
    if not f.exists():
        print(t, "missing"); continue
    rs = [r for r in map(json.loads, open(f)) if r["id"] in items]
    ok = [int(np.argmax(r["lp"])) == items[r["id"]]["answer"] for r in rs]
    print(f"{t}: {100 * np.mean(ok):.1f}% right first (n={len(ok)})")
PY
echo "== matched items (the encoder comparison's 19,093), fcr beside decider on v5"
DEC_TAGS=r231-dv5-s0,r231-dv5-s1 ENCS=li_r227_fcr scripts/chains/r218_owner.sh | grep -E "^\| (all|first-time) "
echo "== r231 owner done $(date -u +%H:%M:%S)"
