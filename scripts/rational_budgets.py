"""Row 220 (owner, 2026-10-06: "make a 'perfectly rational' example too, where there are just a few obvious categories, and the user always
categorizes appropriately, restaurants to restaurants, mortgage payment to mortgage, etc. to see if it can score 100% on a perfectly
rational example, and if not, why not"). Synthetic households where every merchant kind always goes to one obvious category: ten
categories, no misfiles, trips, people, phases or alternation. Merchants are real names from realstyle_merchants_v1 (Overture; open
licences) from its held-out "test" split, so no model trained on our households has seen them. Each household keeps a few regular
merchants per kind (habits: weekly groceries, a monthly mortgage, ...) and sometimes tries a new one (first-time payees). One year.
Two levels of the same transactions: LEVEL=clean (the merchant's plain name, as a cleaned payee) and LEVEL=bank (a fresh statement string
per transaction from statements.render_v2, the grammar measured on a real budget). Budgets are YNAB-shaped for hist_encoder.events.
Expected: payees filed before ~100% for any model that reads its history; first-time payees bounded by what the model knows of the name.
usage: uv run python scripts/rational_budgets.py   (prints counts and examples)
"""
import datetime as dt
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from ai_experiments import statements  # noqa: E402
from ai_experiments.paths import PROCESSED  # noqa: E402

# category -> (merchant kinds, visits per week for the household's habit, regular merchants per kind)
CATS = {
    "Groceries": (["grocery"], 1.5, 2),
    "Restaurants": (["restaurant", "fast_food", "coffee_bakery", "bar"], 3.0, 3),
    "Gas": (["gas"], 1.0, 2),
    "Mortgage": (["rent"], None, 1),        # monthly, the 1st
    "Utilities": (["utility", "phone"], None, 1),  # monthly each, mid-month
    "Health": (["pharmacy", "medical"], 0.3, 1),
    "Shopping": (["big_box", "clothing", "electronics", "home_improvement", "furniture"], 1.0, 2),
    "Car": (["car_repair", "parking"], 0.3, 1),
    "Fun": (["entertainment", "hobby", "events"], 0.5, 1),
    "Pets": (["pet"], 0.3, 1),
}
AMT = {"Groceries": (40, 180), "Restaurants": (8, 70), "Gas": (25, 70), "Mortgage": (1400, 2600), "Utilities": (40, 160), "Health": (10, 120),
       "Shopping": (15, 250), "Car": (10, 400), "Fun": (12, 120), "Pets": (15, 90)}
N, EXPLORE, D0 = 50, 0.15, dt.date(2025, 1, 1)


def _pool():
    by = {}
    for m in json.loads((PROCESSED / "realstyle_merchants_v1.json").read_text())["merchants"]:
        if m["split"] == "test":
            by.setdefault(m["kind"], []).append(m["name"])
    return by


def budgets(level="clean", n=N):
    pool, out = _pool(), []
    for h in range(n):
        rng = random.Random(220_000 + h)
        reg = {(c, k): rng.sample(pool[k], r) for c, (ks, _, r) in CATS.items() for k in ks}
        cats = [dict(id=f"c{i}", name=c, category_group_id="g", hidden=False, deleted=False) for i, c in enumerate(CATS)]
        cid = {c: f"c{i}" for i, c in enumerate(CATS)}
        rows = []
        for day in range(365):
            d = D0 + dt.timedelta(days=day)
            for c, (ks, per_week, _) in CATS.items():
                if per_week is None:  # monthly bills: one per kind on a fixed day
                    for j, k in enumerate(ks):
                        if d.day == (1 if c == "Mortgage" else 12 + 3 * j):
                            rows.append((d, c, reg[(c, k)][0], k))
                    continue
                if rng.random() < per_week / 7:
                    k = rng.choice(ks)
                    name = rng.choice(pool[k]) if rng.random() < EXPLORE else rng.choice(reg[(c, k)])
                    rows.append((d, c, name, k))
        payees, txs, fixed, met = {}, [], {}, set()  # the mortgage is the same amount every month; met: merchants already bought from
        for i, (d, c, name, k) in enumerate(rows):
            text = name if level == "clean" else statements.render_v2(name, rng)
            pid = payees.setdefault(text, f"p{len(payees)}")
            lo, hi = AMT[c]
            amt = fixed.setdefault(name, round(rng.uniform(lo, hi), 2)) if c == "Mortgage" else round(lo * (hi / lo) ** rng.random(), 2)
            txs.append(dict(id=f"r-{h}-{i:05d}", date=d.isoformat(), amount=-round(amt * 1000), category_id=cid[c], payee_id=pid, approved=True,
                            deleted=False, reason=["known" if name in met else "new"],  # scoring segments only, never model input
                            kind=k))  # the merchant's kind: model input only through li_decider's KINDLINE (a simulated lookup)
            met.add(name)
        out.append(dict(id=f"rational-{level}-{h}", category_groups=[dict(id="g", name="Spending")], categories=cats,
                        payees=[dict(id=p, name=t) for t, p in payees.items()], transactions=txs, subtransactions=[]))
    return out


if __name__ == "__main__":
    import hist_encoder as H
    for level in ("clean", "bank"):
        bs = budgets(level)
        ev = [H.events(b) for b in bs]
        n = sum(map(len, ev))
        print(f"{level}: {len(bs)} budgets, {n} outflows, {sum(not x['seen'] for e in ev for x in e)} first-time payee strings; e.g. "
              + "; ".join(f"{x['text']} -> {x['label']}" for x in ev[0][100:104]))
