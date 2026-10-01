"""Tables for PLAN step 83: misleading merchant names (build_mislead.py, mislead_v1), fold 0's held-out users. Each reader before and
after training with the set's in-DB merchants in its database episodes (MISLEAD=mislead_v1).

  M.1  top-1 by merchant kind (misleading / neutral twin / real brand), whether it is in the fact DB, and condition (alone / in the
       user's history), with the share of answers pulled to the category the name suggests (items where that is another option)
usage: uv run python scripts/mislead_tables.py
"""
import json
import warnings

import numpy as np

from ai_experiments import real6_cells as RC
from ai_experiments.paths import PROCESSED

D = "real6_dm_decider_decider_decider-{s}_none_h100bf16st800_f0_ren50_dbep50{ml}_aux100_lora_mislead_v1.noctx.jsonl"
Q = "real6_categoriser_Qwen3.5-2B_none_h100bf16_hf_f0_{a}_lora_mislead_v1_hfs.noctx.jsonl"
READERS = [("decider-4B (row 81), set not trained", D.format(s="4b", ml="")), ("decider-4B, set's DB trained", D.format(s="4b", ml="_mislead_v1")),
           ("decider-2B (row 81), set not trained", D.format(s="2b", ml="")), ("decider-2B, set's DB trained", D.format(s="2b", ml="_mislead_v1")),
           ("Qwen3.5-2B per-option, DB (row 78), set not trained", Q.format(a="alllab_dbep50")),
           ("Qwen3.5-2B per-option, DB + rename, set's DB trained", Q.format(a="ren50_alllab_dbep50_mislead_v1"))]
CELLS = [("misleading, in DB, alone", "misleading", True, "none"), ("neutral twin, in DB, alone", "neutral", True, "none"),
         ("misleading, not in DB, alone", "misleading", False, "none"), ("neutral twin, not in DB, alone", "neutral", False, "none"),
         ("misleading, not in DB, in history", "misleading", False, "hist"), ("neutral twin, not in DB, in history", "neutral", False, "hist"),
         ("real brand, alone", "real", False, "none")]

if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    items = {i["id"]: i for i in json.loads((PROCESSED / "mislead_v1.json").read_text())["items"]}
    print("**Table M.1: misleading merchant names, fold 0: top-1 % / share pulled to the name's category % (n; n with a separate decoy option)**\n")
    print("| reader | " + " | ".join(c[0] for c in CELLS) + " |"); print("|---|" + "---|" * len(CELLS))
    for lab, f in READERS:
        r = RC.load_recs(f)
        if not r:
            continue
        out = []
        for _, kind, in_db, cond in CELLS:
            ids = [i for i in r if i in items and items[i]["mkind"] == kind and items[i]["in_db"] == in_db and items[i]["cond"] == cond]
            if not ids:
                out.append("–"); continue
            pred = {i: int(np.argmax(r[i]["sum_lp"])) for i in ids}
            acc = 100 * np.mean([pred[i] == items[i]["answer"] for i in ids])
            dec = [i for i in ids if items[i]["decoy_idx"] >= 0]
            pull = 100 * np.mean([pred[i] == items[i]["decoy_idx"] for i in dec]) if dec else float("nan")
            out.append(f"{acc:.1f} / {pull:.0f} ({len(ids)}; {len(dec)})")
        print(f"| {lab} | " + " | ".join(out) + " |")
