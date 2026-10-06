#!/bin/bash
# Row 232: the Ettin-1B option scorer (§104 / §110's encoder recipe, seed 0) read on the owner's budget on the local GPU, from decider's own
# items (real_budget_eval's items_grp_sim2.json, the split prompts decider read): scores stay beside the items, only aggregates printed;
# then % right first beside decider (v4 tuned r190-g, v5 r231-dv5) on the same items, overall and first-time payees.
set -u
cd "$(dirname "$0")/../.."
BUDGET=${BUDGET:-96c06c41-f26f-4a44-98ed-eaba2f471e1e}
D="$HOME/.local/share/ynab-real-eval/$BUDGET"
ENC=${ENC:-encmask_ettin1b_st3000_h100fresh_f0_decnone_h100fresh_emp20_f0_ren50_dbep50_mislead_v1_alt10s_lk10_ov10_odb20_short15_kinds20_ev10soft}
TAG=${TAG:-enc1b-s0}
echo "== $TAG $(date -u +%H:%M:%S)"
MAXLEN=${MAXLEN:-2048} PRIVATE_EVERY=${EVERY:-1} ARCH=mask INIT=ettin1b DEC_EPISODES=1 STEPS=0 LOAD_FROM=$ENC PRIVATE_ITEMS="$D/items_grp_sim2.json" \
  PRIVATE_OUT="$D/scores_${TAG}_split_grp_sim2.jsonl" uv run python scripts/exp_encoder_mask.py 2>&1 | grep -E "^ +[0-9]+|items in|Error|error" | tail -20
TAGS="r190-g-s0,r190-g-s1,r231-dv5-s0,r231-dv5-s1,$TAG" BUDGET=$BUDGET PYTHONPATH=scripts uv run python - <<'PY'
import json, os, numpy as np, real_budget_eval as RB
items = {it["id"]: it for it in json.loads((RB.OUT / "items_grp_sim2.json").read_text())["items"] if it["answer"] >= 0}
first = {i for i, it in items.items() if it.get("payee_seen") is False}
sc = {}
for t in os.environ["TAGS"].split(","):
    f = RB.OUT / f"scores_{t}_split_grp_sim2.jsonl"
    if f.exists():
        sc[t] = {r["id"]: int(np.argmax(r["lp"])) == items[r["id"]]["answer"] for r in map(json.loads, open(f)) if r["id"] in items}
common = set.intersection(*map(set, sc.values()))
print(f"| reader | all (n={len(common)}) | first-time payees (n={len(common & first)}) |\n|---|---|---|")
for t, ok in sc.items():
    print(f"| {t} | {100 * np.mean([ok[i] for i in common]):.1f} | {100 * np.mean([ok[i] for i in common & first]):.1f} |")
PY
echo "== r232 owner done $(date -u +%H:%M:%S)"
