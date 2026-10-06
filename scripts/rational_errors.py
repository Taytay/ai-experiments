"""Row 220: why a model misses on the perfectly rational households. Per error on a merchant bought before (and a sample on new
merchants): the query, the gold and chosen categories with their probabilities, whether the merchant's earlier filing is among the query's
neighbours and in the gold category's document; then counts of each cause. Synthetic data only (safe to print).
usage: READ=rational_clean ARMS=li_r217_p3 [N=10] uv run python scripts/rational_errors.py
"""
import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import li_decider as L  # noqa: E402
import rational_budgets as R  # noqa: E402

level = os.environ.get("READ", "rational_clean").split("_", 1)[1]
bs = R.budgets(level, int(os.environ.get("N", "10")))
for arm in os.environ["ARMS"].split(","):
    run = L.load(arm)
    cause, shown = Counter(), 0
    for b in bs:
        mark = {t["id"]: t["reason"][0] for t in b["transactions"]}
        clean = {t["id"]: t for t in b["transactions"]}
        ev, sco = run(b)
        for i, (e, sc) in enumerate(zip(ev, sco)):
            ids = list(sc); v = np.array([sc[c] for c in ids]); p = np.exp(v - v.max()); p /= p.sum()
            top = ids[int(np.argmax(p))]
            if top == e["gold"] or mark[e["id"]] != "known":
                continue
            same = [j for j in range(i) if ev[j]["payee"] == e["payee"]]  # earlier filings with the identical string
            nb = list(e.get("nb", []))[:L.NB]
            in_nb = any(ev[j]["payee"] == e["payee"] for j in nb)
            in_doc = e["payee"] in e["d"][e["gold"]]
            k = ("identical string filed before" if same else "string never seen (bank rendering)") + ("; in the query's neighbours" if in_nb else "; not in neighbours") + ("; in gold's document" if in_doc else "")
            cause[k] += 1
            if shown < 12:
                shown += 1
                print(f"- {e['q'][:230]!r}\n    gold {e['labels'][e['gold']]} {p[ids.index(e['gold'])]:.2f}; chose {e['labels'][top]} {p.max():.2f}; {k}")
    print(f"\n{run.name}: causes of errors on merchants bought before: {dict(cause)}\n")
