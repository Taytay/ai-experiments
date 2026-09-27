"""Tables for PLAN step 73 (POI-1 unseen kinds and category meaning; REPORT 64.2 hypothesis 3): a header describing each category by the
kinds the user has filed under it (poi1_v1_desc*), unseen-kind training episodes (POI_UNSEEN), and an instruction line for untrained
readers (poi1_v1_instr*). Same readers' recipe as row 72 (3B, fold 0 held out, 800 steps, all-label loss, rename 0.5).

  D.1  the scorecard on fold 0's held-out users, reader x layout, with seen / unseen kind top-1
  D.2  unseen-kind items by category name type (readable / merged / coined)
usage: uv run python scripts/poi1_desc_tables.py
"""
import sys
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import poi1_kinds_tables as K  # noqa: E402
import poi1_tables as PT  # noqa: E402

K.LAYOUT.update({"desc": ("poi1_v1_desc", "noctx"), "desc + kinds": ("poi1_v1_desc_kinds", "noctx"),
                 "desc + kshots + kinds": ("poi1_v1_desc_kshots_kinds", "noctx"), "instruction": ("poi1_v1_instr", "noctx"),
                 "instruction + kshots + kinds": ("poi1_v1_instr_kshots_kinds", "noctx")})
C = K.CAT
UNTRAINED = ["plain", "instruction", "desc", "kshots + kinds", "instruction + kshots + kinds", "desc + kinds", "desc + kshots + kinds"]
READERS = [("Qwen2.5-3B-Instruct, untrained", "real6_Qwen2.5-3B-Instruct", UNTRAINED),
           ("Qwen2.5-14B-Instruct, untrained", "real6_Qwen2.5-14B-Instruct", UNTRAINED),
           ("3B trained with kinds (row 72)", f"{C}_pkind_f0_ren50_alllab_lora", ["kinds", "kshots + kinds", "desc + kinds", "desc + kshots + kinds"]),
           ("3B trained with kinds, unseen episodes", f"{C}_pkind_uns50_f0_ren50_alllab_lora", ["kinds", "kshots + kinds"]),
           ("3B trained with desc, unseen episodes", f"{C}_pdesc_uns50_f0_ren50_alllab_lora", ["desc"]),
           ("3B trained with desc + kinds", f"{C}_pkind_pdesc_f0_ren50_alllab_lora", ["desc + kinds", "desc + kshots + kinds"]),
           ("3B trained with desc + kinds, unseen episodes", f"{C}_pkind_pdesc_uns50_f0_ren50_alllab_lora", ["desc + kinds", "desc + kshots + kinds"])]

if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    items, users = PT.load()
    print("**Table D.1: POI-1 fold 0's held-out users, the scorecard by reader and layout (columns as Table 61.1)**\n")
    print(PT.HEAD.replace("| reader |", "| reader | layout |") + " seen-kind top-1 | unseen-kind top-1 |")
    print("|" + "---|" * 15)
    for label, prefix, lays in READERS:
        for lay in lays:
            recs = K.recs_of(prefix, lay, items)
            if not recs:
                continue
            sc = PT.card(recs, items, users, True)
            seen = 100 * np.mean([K.ok(r) for i, r in recs.items() if items[i]["seen"]])
            unseen = 100 * np.mean([K.ok(r) for i, r in recs.items() if not items[i]["seen"]])
            print(PT.row(label, sc).replace(f"| {label} |", f"| {label} | {lay} |", 1) + f" {seen:.1f} | {unseen:.1f} |")
    nts = ("standard", "renamed", "new")
    f0 = [i for i in items if items[i]["user"] % 4 == 0 and not items[i]["seen"]]
    print("\n**Table D.2: unseen-kind items, top-1 % by category name type, fold 0**\n")
    print("| reader | layout | " + " | ".join(f"{nt} (n={sum(items[i]['name_type'] == nt for i in f0)})" for nt in nts) + " |")
    print("|---|---|---|---|---|")
    for label, prefix, lays in READERS:
        for lay in lays:
            recs = K.recs_of(prefix, lay, items)
            if recs:
                print(f"| {label} | {lay} | " + " | ".join(f"{100 * np.mean([K.ok(r) for i, r in recs.items() if not items[i]['seen'] and items[i]['name_type'] == nt]):.0f}" for nt in nts) + " |")
