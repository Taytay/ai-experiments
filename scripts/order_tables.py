"""Tables for PLAN step 50 (EVAL-8), re-scoped to the one-slot readers that lead after REPORT 69: how much the answer depends on the
order the options are shown in. Each reader scored in its original order and with each item's options shuffled under three seeds
(exp_decision_models.py ORDER_SEED; the category list in the prompt is unchanged, only the labelled options move). Fold 0's users.

  R.1  per reader: top-1 in each order, the share of items whose answer changes in any order (flip rate), and the top-1 of the
       four-order average (mean log-probability per option)
usage: uv run python scripts/order_tables.py
"""
import json

import numpy as np

from ai_experiments import real6 as R6
from ai_experiments import real6_cells as RC
from ai_experiments.paths import PROCESSED

READERS = [("decider-2B one slot, POI-1", "real6_dm_decider_decider_decider-2b_poi1_v1_none_h100bf16_pkind_pdesc_uns50_f0_ren50_lora_poi1_v1_desc_kshots_kinds{o}.noctx.jsonl", "poi"),
           ("Qwen3.5-2B one slot, POI-1", "real6_dm_decider_slot_Qwen3.5-2B_poi1_v1_none_h100bf16_pkind_pdesc_uns50_f0_ren50_lora_poi1_v1_desc_kshots_kinds{o}.noctx.jsonl", "poi"),
           ("decider-2B one slot + shot labels, REAL-6", "real6_dm_decider_decider_decider-2b_none_h100bf16st800_f0_dbep50_aux100_lora_real6{o}.noctx.jsonl", "r6"),
           ("Qwen3.5-2B one slot + shot labels, REAL-6", "real6_dm_decider_slot_Qwen3.5-2B_none_h100bf16st800_f0_dbep50_aux100_lora_real6{o}.noctx.jsonl", "r6")]

if __name__ == "__main__":
    items = {"poi": {i["id"]: i for i in json.loads((PROCESSED / "poi1_v1.json").read_text())["items"]}, "r6": {i["id"]: i for i in R6.load()["items"]}}
    print("**Table R.1: one-slot readers under four option orders (the original and three seeds), fold 0: top-1 per order, flip rate "
          "(items whose top-1 differs in any order), and the four-order average**\n")
    print("| reader | n | original | seed 1 | seed 2 | seed 3 | spread (points) | flip rate | four-order average |"); print("|---|---|---|---|---|---|---|---|---|")
    for lab, pat, s in READERS:
        runs = [RC.load_recs(pat.format(o=o)) for o in ("", "_ord1", "_ord2", "_ord3")]
        if not all(runs):
            continue
        ids = sorted(set.intersection(*(set(r) for r in runs)) & {i for i in items[s] if items[s][i]["user"] % 4 == 0})
        preds = np.array([[int(np.argmax(r[i]["sum_lp"])) for i in ids] for r in runs]); gold = np.array([runs[0][i]["answer"] for i in ids])
        acc = [100 * (p == gold).mean() for p in preds]
        flip = 100 * (preds != preds[0]).any(0).mean()
        avg = np.array([int(np.argmax(np.mean([np.asarray(r[i]["sum_lp"]) for r in runs], 0))) for i in ids])
        print(f"| {lab} | {len(ids)} | " + " | ".join(f"{a:.1f}" for a in acc) + f" | {max(acc) - min(acc):.1f} | {flip:.1f}% | {100 * (avg == gold).mean():.1f} |")
