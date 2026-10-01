"""Tables for PLAN step 74 (REAL-20; REPORT 64.2 hypothesis 4): label-induction training with decoys and empty categories. Qwen3.5-2B
trained on REAL-6 users of fold 0 with rename augmentation (row 64's recipe: all-label loss, 200 steps) and exp_categoriser.py's EMPTY=3
(coined categories with no examples) and DECOY=0.3 (a same-kind shot filed under a category that holds none of its kind, its label
unsupervised), scored on label induction v1 / v2 (all 300 queries per condition) against each condition's ceiling
(ai_experiments.ceiling.label_induction), and on REAL-6 fold 0.

  T.1  v2 top-1 % by condition (ceiling per condition), with each reader's share of the whole set's ceiling
  T.2  the same on v1
  T.3  REAL-6 fold 0 top-1 (does the augmentation cost the ordinary task?)
usage: uv run python scripts/li_training_tables.py
"""
import json

import numpy as np

from ai_experiments import ceiling as CE
from ai_experiments import real6 as R6
from ai_experiments import real6_cells as RC
from ai_experiments.paths import PROCESSED

T = "real6_categoriser_Qwen3.5-2B_none_h100bf16_hf{s}_f0_ren50_alllab_lora"
READERS = [("Qwen3.5-2B untrained", "real6_Qwen3.5-2B_label_induction_{v}_hfs.noctx.jsonl", None),
           ("rename", T.format(s="") + "_label_induction_{v}_hfs.noctx.jsonl", T.format(s="") + "_hfs.noctx.jsonl"),
           ("rename + empty", T.format(s="_emp3") + "_label_induction_{v}_hfs.noctx.jsonl", T.format(s="_emp3") + "_hfs.noctx.jsonl"),
           ("rename + decoys", T.format(s="_dec30") + "_label_induction_{v}_hfs.noctx.jsonl", T.format(s="_dec30") + "_hfs.noctx.jsonl"),
           ("rename + decoys + empty", T.format(s="_dec30_emp3") + "_label_induction_{v}_hfs.noctx.jsonl", T.format(s="_dec30_emp3") + "_hfs.noctx.jsonl")]


def ok(r):
    return int(np.argmax(r["sum_lp"])) == r["answer"]


if __name__ == "__main__":
    for k, v in enumerate(("v2", "v1")):
        items = {i["id"]: i for i in json.loads((PROCESSED / f"label_induction_{v}.json").read_text())["items"]}
        conds = list(dict.fromkeys(i["condition"] for i in items.values()))
        recs = [RC.load_recs(f.format(v=v)) for _, f, _ in READERS]
        print(f"**Table T.{k + 1}: label induction {v}, top-1 % by condition (300 queries each; Qwen3.5-2B)**\n")
        print("| condition | ceiling | " + " | ".join(lab for lab, _, _ in READERS) + " |"); print("|---|---|" + "---|" * len(READERS))
        for c in conds + ["all conditions"]:
            ids = [i for i in items if c == "all conditions" or items[i]["condition"] == c]
            ce = [CE.label_induction(items[i]) for i in ids]
            cells = [f"{100 * np.mean([ok(r[i]) for i in ids]):.1f}" + (f" ({CE.share([ok(r[i]) for i in ids], ce):.0f}% of ceiling)" if c == "all conditions" else "") for r in recs]
            print(f"| {c} | {100 * np.mean(ce):.0f} | " + " | ".join(cells) + " |")
        print()
    r6 = {i["id"]: i for i in R6.load("v1")["items"]}
    print("**Table T.3: REAL-6, fold 0's held-out users (298 items), top-1 %**\n")
    print("| reader | top-1 |"); print("|---|---|")
    for lab, _, f in READERS[1:]:
        r = {i: x for i, x in RC.load_recs(f).items() if r6[i]["user"] % 4 == 0}
        print(f"| {lab} | {100 * np.mean([ok(x) for x in r.values()]):.1f} |")
