"""Row 234: where a reader's first purchases at a merchant go wrong on the rational households (RATIONAL_PAYEES / RATIONAL_CATS as
rational_budgets.py): per merchant kind, the share right first and the categories chosen instead. Synthetic data; ARM (li_r227_fcr)."""
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import li_decider as LD  # noqa: E402
import rational_budgets as RG  # noqa: E402

run = LD.load(os.environ.get("ARM", "li_r227_fcr"))
by_kind, wrong = defaultdict(list), defaultdict(Counter)
for b in RG.budgets("bank"):
    kind = {t["id"]: t["kind"] for t in b["transactions"]}
    new = {t["id"] for t in b["transactions"] if t["reason"][0] == "new"}
    names = {c["id"]: c["name"] for c in b["categories"]}
    base = {c["id"]: c for c in RG.CATS} if False else None
    ev, sco = run(b)
    for e, sc in zip(ev, sco):
        if e["id"] not in new or e["gold"] not in sc:
            continue
        pred = max(sc, key=sc.get)
        by_kind[kind[e["id"]]].append(pred == e["gold"])
        if pred != e["gold"]:
            gi, pi = int(e["gold"][1:]), int(pred[1:])  # category ids c0..c9 in rational_budgets.CATS order
            wrong[kind[e["id"]]][f"{list(RG.CATS)[gi]} -> {list(RG.CATS)[pi]}"] += 1
print("| kind | first purchases | right first | most common misses (true -> chosen) |\n|---|---|---|---|")
for k, v in sorted(by_kind.items(), key=lambda x: np.mean(x[1])):
    print(f"| {k} | {len(v)} | {100 * np.mean(v):.0f} | " + "; ".join(f"{m} {n}" for m, n in wrong[k].most_common(3)) + " |")
