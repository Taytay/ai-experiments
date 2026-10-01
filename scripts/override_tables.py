"""Tables for PLAN step 85 (REAL-15): the user's own filing against the database and the name (override_v1, build_override.py). Two of
the user's shots file a fact-DB merchant under the user's category for its DB category (agree) or under another category (override,
which is then the answer). decider-4B before (row 81, and row 83 with the misleading names in its DB) and trained with row 81's recipe +
MISLEAD + random A..Z labels, without (ctl) and with lookup / override episodes (LOOKUP=OVERRIDE=0.1), two seeds each (mean [range]).

  V.1  top-1 by merchant kind and condition; on override items also the share pulled back to the DB category; REAL-6 top-1
usage: uv run python scripts/override_tables.py
"""
import json
import warnings

import numpy as np

from ai_experiments import real6 as R6
from ai_experiments import real6_cells as RC
from ai_experiments.paths import PROCESSED

B = "real6_dm_decider_decider_decider-4b_none_{tag}_f0_ren50_dbep50{ml}{sfx}_aux100{lab}_lora_{s}{rd}.noctx.jsonl"
READERS = [("row 81 (set's merchants not in its DB)", [dict(tag="h100bf16st800", ml="", sfx="", lab="", rd="")]),
           ("row 83 (misleading names in its DB)", [dict(tag="h100bf16st800", ml="_mislead_v1", sfx="", lab="", rd="")]),
           ("control: + random labels", [dict(tag=t, ml="_mislead_v1", sfx="", lab="_labrand26", rd="_labrand26") for t in ("h100bf16st800", "h100bf16st800s1")]),
           ("+ lookup and override episodes", [dict(tag=t, ml="_mislead_v1", sfx="_lk10_ov10", lab="_labrand26", rd="_labrand26") for t in ("h100bf16st800", "h100bf16st800s1")])]
KINDS = [("REAL-6 DB-only chain", "real6_known"), ("REAL-6 DB-only opaque", "real6_opaque"), ("misleading name", "misleading"), ("neutral twin", "neutral")]


def fmt(v):
    return "–" if not v else f"{np.mean(v):.1f}" + (f" [{min(v):.1f}, {max(v):.1f}]" if len(v) > 1 else "")


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    items = {i["id"]: i for i in json.loads((PROCESSED / "override_v1.json").read_text())["items"]}
    r6 = {i["id"]: i for i in R6.load()["items"]}
    cols = [f"{k}: {c}" for k, _ in KINDS for c in ("agree", "override", "override, pulled to DB")] + ["REAL-6 top-1"]
    print("**Table V.1: the user's filing (2 shots) against the database and the name, fold 0: top-1 % and the share of override items "
          "answered with the DB category instead**\n")
    print("| reader | " + " | ".join(cols) + " |"); print("|---|" + "---|" * len(cols))
    for lab, runs in READERS:
        vals = {c: [] for c in cols}
        for a in runs:
            r = RC.load_recs(B.format(s="override_v1", **a))
            if not r:
                continue
            pred = {i: int(np.argmax(v["sum_lp"])) for i, v in r.items() if i in items}
            for kname, kind in KINDS:
                for cond in ("agree", "override"):
                    ids = [i for i in pred if items[i]["mkind"] == kind and items[i]["cond"] == cond]
                    if ids:
                        vals[f"{kname}: {cond}"].append(100 * np.mean([pred[i] == items[i]["answer"] for i in ids]))
                        if cond == "override":
                            vals[f"{kname}: override, pulled to DB"].append(100 * np.mean([pred[i] == items[i]["db_idx"] for i in ids]))
            rr = RC.load_recs(B.format(s="real6", **a))
            if rr:
                f0 = [i for i in rr if r6[i]["user"] % 4 == 0]
                vals["REAL-6 top-1"].append(100 * np.mean([int(np.argmax(rr[i]["sum_lp"])) == rr[i]["answer"] for i in f0]))
        if any(vals.values()):
            print(f"| {lab} | " + " | ".join(fmt(v) for v in vals.values()) + " |")
