"""Tables for PLAN rows 166-167 on the owner's real budget (real_budget_eval.py's private items and scores; aggregates only): every
complete scores_<arm>.jsonl in OUT against YNAB's rule, the rule with payee resolution (aliases: real_budget_eval.py SIM=2), by
group: right first, top 3 / 5 / 10, MRR, the top suggestion being a category ever used for the payee (any time, future included), and
the probability left on categories never used for the payee (owner, 2026-10-03: "knowing which payees something might be vs
DEFINITELY isn't").
env: BUDGET, OUT (as real_budget_eval.py), ARMS (comma list of arm names; default every complete one).
usage: BUDGET=<id> uv run python scripts/real_budget_tables.py
"""
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np

BUDGET = os.environ["BUDGET"]
OUT = Path(os.environ.get("OUT", Path.home() / ".local" / "share" / "ynab-real-eval" / BUDGET))


def payee(it):
    return it["prompt"].rsplit("Transaction: ", 1)[1].split(" | ")[1]


if __name__ == "__main__":
    items = {it["id"]: it for it in json.loads((OUT / "items.json").read_text())["items"] if it["answer"] >= 0}
    alias_items = OUT / "items_sim2.json"
    ral = {it["id"]: it for it in json.loads(alias_items.read_text())["items"]} if alias_items.exists() else {}
    ever = defaultdict(set)
    for it in items.values():
        ever[payee(it)].add(it["gold"])
    arms = {}
    for f in sorted(OUT.glob("scores_*.jsonl")):
        name = f.stem[len("scores_"):]
        recs = {r["id"]: np.asarray(r["lp"]) for r in map(json.loads, open(f))}
        if len(recs) >= len(items):
            arms[name] = recs
    if os.environ.get("ARMS"):
        arms = {k: v for k, v in arms.items() if k in os.environ["ARMS"].split(",")}
    groups = {"all": list(items), "payee filed before": [i for i in items if items[i]["payee_seen"]],
              "first-time payee": [i for i in items if not items[i]["payee_seen"]]}
    if ral:
        groups["first-time payee, but an alias filed before"] = [i for i in groups["first-time payee"] if ral[i]["seen_alias"]]
        groups["first-time payee, no alias either"] = [i for i in groups["first-time payee"] if not ral[i]["seen_alias"]]
    for g, ids in groups.items():
        print(f"\n**{g} (n={len(ids)}): right first / top 3 / top 5 / top 10 / MRR / top-1 a category ever used for the payee / probability on never-used categories**\n")
        print("| reader | 1st | top 3 | top 5 | top 10 | MRR | top-1 ever used | p(never used) |"); print("|---|---|---|---|---|---|---|---|")
        print(f"| YNAB's rule | {100 * np.mean([items[i]['rule'] == items[i]['gold'] for i in ids]):.1f} | | | | | | |")
        if ral:
            print(f"| YNAB's rule with payee resolution (aliases) | {100 * np.mean([ral[i]['rule_alias'] == items[i]['gold'] for i in ids]):.1f} | | | | | | |")
        for name, S in arms.items():
            rk = np.array([1 + int((S[i] > S[i][items[i]["answer"]]).sum() + (S[i] == S[i][items[i]["answer"]]).sum() - 1) for i in ids])
            ev = [{items[i]["options"].index(c) for c in ever[payee(items[i])] if c in items[i]["options"]} for i in ids]
            top1_ever = np.mean([int(np.argmax(S[i])) in e for i, e in zip(ids, ev)])
            never = np.mean([1 - float(np.exp(S[i][sorted(e)]).sum()) for i, e in zip(ids, ev)])
            print(f"| {name} | {100 * np.mean(rk == 1):.1f} | {100 * np.mean(rk <= 3):.1f} | {100 * np.mean(rk <= 5):.1f} | {100 * np.mean(rk <= 10):.1f} | "
                  f"{np.mean(1 / rk):.3f} | {100 * top1_ever:.1f} | {100 * never:.1f}% |")
