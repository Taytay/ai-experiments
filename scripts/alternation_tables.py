"""Tables for PLAN step 86: users who alternate between categories at one merchant (alternation_v1, build_alternation.py), fold 0.
Readers: row 81's recipe (three seeds, letters), row 84's control (random A..Z labels, two seeds) and the same trained with alternation
episodes (ALT=0.15, two seeds); mean over seeds [range].

  A.1  per merchant kind and scenario: top-1 against the cell's ceiling (the rule's answer), the mean probability on the gold, and
       for the unobservable and control cells the share of items given p >= 0.9 on one category (a confident answer where the truth
       is a 60 / 40 split, or where the specific category is never used for this merchant)
usage: uv run python scripts/alternation_tables.py
"""
import json
import warnings

import numpy as np

from ai_experiments import real6_cells as RC
from ai_experiments.paths import PROCESSED

D = "real6_dm_decider_decider_decider-4b_none_{tag}_f0_ren50_dbep50{alt}_aux100{lab}_lora_alternation_v1{rd}.noctx.jsonl"
READERS = [("row 81 recipe (letters)", [D.format(tag=t, alt="", lab="", rd="") for t in ("h100bf16st800", "h100bf16st800s1", "h100bf16st800s2")]),
           ("+ random labels (row 84 control)", [D.format(tag=t, alt="", lab="_labrand26", rd="_labrand26") for t in ("h100bf16st800", "h100bf16st800s1")]),
           ("+ random labels + alternation episodes", [D.format(tag=t, alt="_alt15", lab="_labrand26", rd="_labrand26") for t in ("h100bf16st800", "h100bf16st800s1")])]
SCEN = ("observable", "partial", "unobservable", "control")


def sm(z):
    z = np.asarray(z, float); e = np.exp(z - z.max()); return e / e.sum()


def fmt(v, d=1):
    return "–" if not v else f"{np.mean(v):.{d}f}" + (f" [{min(v):.{d}f}, {max(v):.{d}f}]" if len(v) > 1 else "")


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    items = {i["id"]: i for i in json.loads((PROCESSED / "alternation_v1.json").read_text())["items"]}
    for kind in ("store", "restaurant"):
        ceil = {s: 100 * np.mean([i["answer"] == i["rule_pred"] for i in items.values() if i["kind"] == kind and i["scen"] == s]) for s in SCEN}
        print(f"**Table A.1 ({kind}s; rule: {'amount >= $40' if kind == 'store' else 'Fri / Sat'} to the specific category): top-1 % "
              f"(ceiling), mean P(gold), and p >= 0.9 share where a confident answer is wrong in kind**\n")
        cols = [f"{s}: top-1 (ceiling {ceil[s]:.0f})" for s in SCEN] + [f"{s}: P(gold)" for s in SCEN] + ["unobservable: p >= 0.9", "control: specific category chosen"]
        print("| reader | " + " | ".join(cols) + " |"); print("|---|" + "---|" * len(cols))
        for lab, files in READERS:
            vals = {c: [] for c in cols}
            for f in files:
                r = RC.load_recs(f)
                if not r:
                    continue
                for s in SCEN:
                    ids = [i for i in r if i in items and items[i]["kind"] == kind and items[i]["scen"] == s]
                    p = {i: sm(r[i]["sum_lp"]) for i in ids}
                    vals[cols[SCEN.index(s)]].append(100 * np.mean([int(np.argmax(p[i])) == items[i]["answer"] for i in ids]))
                    vals[cols[4 + SCEN.index(s)]].append(float(np.mean([p[i][items[i]["answer"]] for i in ids])))
                    if s == "unobservable":
                        vals[cols[8]].append(100 * np.mean([p[i].max() >= 0.9 for i in ids]))
                    if s == "control":
                        vals[cols[9]].append(100 * np.mean([int(np.argmax(p[i])) == items[i]["b_idx"] for i in ids]))
            if any(vals.values()):
                print(f"| {lab} | " + " | ".join(fmt(v, 2 if "P(gold)" in c else 1) for c, v in vals.items()) + " |")
        print()
