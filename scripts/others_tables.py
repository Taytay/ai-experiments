"""Tables for PLAN step 124 (REAL-23): a line before the query naming the categories other users file the payee under ("Other users file
this payee as: Eating out (14), Restaurants (9), ..."; blind_v1_others, built from the other blind_v1 users' filings before the query
date, on 1,040 of 1,500 items). decider-4B with labelled rows: the recipe (read zero-shot with the line) and the recipe trained with the
line in half its episodes (OTHERS=0.5, two seeds), each read with and without the line. Top-1 by what decides the item, and effort
(the scorecard's 98% cut at W = 5, and the per-item rule at W = 10, REPORT 118).
usage: uv run python scripts/others_tables.py
"""
import json
import sys
import warnings
from pathlib import Path

import numpy as np

from ai_experiments.paths import PROCESSED

sys.path.insert(0, str(Path(__file__).parent))
from decision_tables import evaluate, fmt, folds_calibrate, load  # noqa: E402

REC = "ren50_dbep50_mislead_v1_alt10s_lk10_ov10"
F = "results/per_item/real6_dm_decider_decider_decider-4b_none_h100bf16st800{s}_emp20_f0_" + REC + "{oth}_aux100_labrand255_laylabelled_shots_ev10soft_lora_{set}_labrand255_laylabelled_shots.noctx.jsonl"
ARMS = [("the recipe", "", "blind_v1", ("", "s1")), ("the recipe, line shown (zero-shot)", "", "blind_v1_others", ("",)),
        ("trained with the line, line not shown", "_oth50", "blind_v1", ("", "s1")), ("trained with the line, line shown", "_oth50", "blind_v1_others", ("", "s1"))]

if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    items = {i["id"]: i for i in json.loads((PROCESSED / "blind_v1_others.json").read_text())["items"]}
    has = {i for i, x in items.items() if "Other users file" in x["prompt"]}
    groups = [("all", lambda x: True), ("items with the line", lambda x: x["id"] in has), ("new users (0-10 rows)", lambda x: x["why"] == "short_history"),
              ("new merchants (chain / descriptive / opaque)", lambda x: x["why"].startswith("new_merchant")),
              ("user's own habit (idiosyncratic, changed mind, named)", lambda x: x["why"] in ("idiosyncratic", "changed_mind", "named_category")),
              ("plain / recurring / income", lambda x: x["why"] in ("plain", "recurring", "income"))]
    print("**Table OT.1: top-1 on blind_v1 by group (mean over seeds)**\n")
    print("| arm | " + " | ".join(f"{g} (n={sum(fn(x) for x in items.values())})" for g, fn in groups) + " |"); print("|---|" + "---|" * len(groups))
    recs = {}
    for name, oth, st, seeds in ARMS:
        recs[name] = [load(F.format(s=s, oth=oth, set=st)) for s in seeds]
        cells = []
        for g, fn in groups:
            ids = [i for i, x in items.items() if fn(x)]
            cells.append(f"{np.mean([100 * np.mean([int(np.argmax(r[i])) == items[i]['answer'] for i in ids if i in r]) for r in recs[name]]):.1f}")
        print(f"| {name} | " + " | ".join(cells) + " |")
    print("\n**Table OT.2: effort on blind_v1 (mean [range] over seeds)**\n")
    print("| arm | 98% cut, W = 5: auto-filed % / effort | per-item, W = 10: auto-filed % / precision % / effort |"); print("|---|---|---|")
    for name, *_ in ARMS:
        a = np.array([evaluate(folds_calibrate(r, items), items, 5, "fixed 98%") for r in recs[name]])
        b = np.array([evaluate(folds_calibrate(r, items), items, 10, "per-item") for r in recs[name]])
        print(f"| {name} | {fmt(a[:, 0])} / {fmt(a[:, 2], 3)} | {fmt(b[:, 0])} / {fmt(b[:, 1])} / {fmt(b[:, 2], 3)} |")
