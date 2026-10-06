"""Row 233: why synthetic payees end up under 2+ categories. For payees with 3+ outflows under 2+ categories, the share of them in which
each generator reason (trip, holiday, move, reorg, misfile, person, phase, ...) put a filing outside the payee's majority category; and
the share of all outflows per reason. Synthetic households only (TEST_SEEDS); env as for the households (RS_V5, RS_V6)."""
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from two_tower import households  # noqa: E402

a, z = map(int, os.environ.get("TEST_SEEDS", "100000-100009").split("-"))
multi = Counter(); nmulti = n3 = 0; allr = Counter(); ntx = 0
for b in households("test", range(a, z + 1)):
    by = defaultdict(list)
    for t in (b.get("budget") or b)["transactions"]:
        if t["amount"] < 0 and t.get("reason"):
            by[t["payee_id"]].append(t); allr[t["reason"][0]] += 1; ntx += 1
    for ts in by.values():
        if len(ts) < 3:
            continue
        n3 += 1
        cats = Counter(t["category_id"] for t in ts)
        if len(cats) < 2:
            continue
        nmulti += 1
        maj = cats.most_common(1)[0][0]
        for r in {t["reason"][0] for t in ts if t["category_id"] != maj}:
            multi[r] += 1
print(f"payees 3+: {n3}, under 2+ categories: {nmulti / n3:.2f}")
print("| reason | share of multi-category payees with an off-majority filing for it | share of outflows |\n|---|---|---|")
for r, c in multi.most_common():
    print(f"| {r} | {c / nmulti:.2f} | {allr[r] / ntx:.3f} |")
