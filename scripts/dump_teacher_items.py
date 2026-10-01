"""PLAN step 123 (MODEL-19): write the recipe's training-episode pool for this env and SEED as an item set a teacher can read
(data/processed/teacher_eps_<SFX>.json, items with prompt / options / answer, user 0), keeping only first-time-payee episodes (the query's
payee key in no row of the episode), where the large model's knowledge of businesses is what the small model lacks. Run with exactly the
env of the training job (exp_decider_finetune.py builds the same pool from the same env and seed).
usage: <training env> uv run python scripts/dump_teacher_items.py
"""
import json
import runpy
import sys

from ai_experiments import oneslot
from ai_experiments.paths import PROCESSED, ROOT

sys.argv = [sys.argv[0]]
G = runpy.run_path(str(ROOT / "scripts" / "exp_decider_finetune.py"), run_name="teacher_dump")
eps = G["episodes"]()
items = []
for k, e in enumerate(eps):
    ctx, opts, gold = e[0], e[1], e[2]
    names, rows, query = oneslot.parse(ctx)
    key = oneslot.payee_key(query)
    if not key or any(c is not None and oneslot.payee_key(f) == key for f, c, _ in rows):
        continue
    items.append(dict(id=f"EP:{k}", user=0, level="EP", prompt=ctx + "\nCategory:", prompt_ctx=ctx + "\nCategory:", options=[" " + o for o in opts], answer=gold))
name = f"teacher_eps_{G['SFX_POOL']}"
(PROCESSED / f"{name}.json").write_text(json.dumps({"name": name, "items": items}))
print(f"wrote {name}.json: {len(items)} first-time-payee episodes of {len(eps)}", flush=True)
