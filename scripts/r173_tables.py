"""Tables for PLAN step 173 (REPORT 161): decider-4B's recipe with real-style household episodes mixed in (REALSTYLE = 0.25 / 0.5,
seed 0) against the recipe itself (seeds 0, 1, 2), right first % on the synthetic sets (per-item files from exp_decision_models.py):
REAL-6, novel names, misleading names, the user's override, blind_v1 / v2 with and without the other-users line; and blind_v1 / v2 by
the item's kind of difficulty (`why`) for the sets that carry it.
usage: uv run python scripts/r173_tables.py
"""
import glob
import json
from collections import defaultdict

import numpy as np

from ai_experiments.paths import PROCESSED

R = "results/per_item/real6_dm_decider_decider_decider-4b_none_h100bf16st{steps}{seed}_emp20_f0_ren50_dbep50_mislead_v1_alt10s_lk10_ov10_oth50_aux100_labrand255_laylabelled_shots_ev10soft{rs}_lora_{set}_labrand255_laylabelled_shots.noctx.jsonl"
SETS = ["real6", "real6_v1_novel", "mislead_v1", "override_v1", "blind_v1", "blind_v1_others", "blind_v2", "blind_v2_others"]
ARMS = {"recipe (3 seeds)": [("", ""), ("s1", ""), ("s2", "")], "recipe + 25% real-style": [("", "_rs25")], "recipe + 50% real-style": [("", "_rs50")],
        "row 176 A2: + 25% real-style v2": [("", "_rs25v2")], "row 176 B: + 25% real-style v2, field block generated": [("", "_rs25v2f_fl")],
        "row 177 C: + 25% real-style v2, row numbers": [("", "_rs25v2i")], "row 177 D: + row numbers and rationales": [("", "_rs25v2r")],
        "row 179 K: A2 + 40% knowledge 5k x26": [("", "_rs25v2_mk40_5k")], "row 180: 5k x13 at 20%": [("", "_rs25v2_mk20_5k13")],
        "row 180: 20k x6.5 at 40%": [("", "_rs25v2_mk40_20k6")], "row 180: A2 at 3200 steps": [("3200:", "_rs25v2")],
        "row 180: 20k x26 at 40%, 3200 steps": [("3200:", "_rs25v2_mk40_20k26")], "row 180: 48k x11 at 40%, 3200 steps": [("3200:", "_rs25v2_mk40_48k11")],
        "row 181 G: Group: Name households": [("", "_rs25v2g")], "row 181 GK: Group: Name + knowledge 5k": [("", "_rs25v2g_mk40_5k")],
        "row 182 G seed 1": [("s1", "_rs25v2g")], "row 182 GK seed 1": [("s1", "_rs25v2g_mk40_5k")],
        "row 182 GK20: Group: Name + 20k x6.5": [("", "_rs25v2g_mk40_20k6")], "row 182 GL: Group: Name + Kind lines": [("", "_rs25v2gkl")],
        "row 182 GLK20: + Kind lines + 20k x6.5": [("", "_rs25v2gkl_mk40_20k6")]}


def acc(f, sel=None):
    recs = [json.loads(l) for l in open(f)]
    recs = [r for r in recs if sel is None or r["id"] in sel]
    return 100 * np.mean([int(np.argmax(r["sum_lp"])) == r["answer"] for r in recs]), len(recs)


if __name__ == "__main__":
    print("| arm | " + " | ".join(SETS) + " |"); print("|---|" + "---|" * len(SETS))
    for arm, runs in ARMS.items():
        cells = []
        for st in SETS:
            vals = [acc(f)[0] for seed, rs in runs for f in glob.glob(R.format(steps=seed.split(":")[0] if ":" in seed else "800", seed=seed.split(":")[-1] if ":" in seed else seed, rs=rs, set=st))]
            cells.append(f"{np.mean(vals):.1f}" + (f" [{min(vals):.1f}-{max(vals):.1f}]" if len(vals) > 1 else "") if vals else "-")
        print(f"| {arm} | " + " | ".join(cells) + " |")
    for st in ("blind_v1", "blind_v2"):
        items = {i["id"]: i for i in json.loads((PROCESSED / f"{st}.json").read_text())["items"]}
        whys = defaultdict(set)
        for i, it in items.items():
            whys[it.get("why", "?")].add(i)
        whys = {w: s for w, s in sorted(whys.items(), key=lambda x: -len(x[1])) if len(s) >= 40}
        print(f"\n**{st} by kind of item, right first %**\n")
        print("| arm | " + " | ".join(f"{w} ({len(s)})" for w, s in whys.items()) + " |"); print("|---|" + "---|" * len(whys))
        for arm, runs in ARMS.items():
            cells = []
            for w, sel in whys.items():
                vals = [acc(f, sel)[0] for seed, rs in runs for f in glob.glob(R.format(steps=seed.split(":")[0] if ":" in seed else "800", seed=seed.split(":")[-1] if ":" in seed else seed, rs=rs, set=st))]
                cells.append(f"{np.mean(vals):.1f}" if vals else "-")
            print(f"| {arm} | " + " | ".join(cells) + " |")
