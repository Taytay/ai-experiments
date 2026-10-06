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
import os
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

# Row 234 (owner, 2026-10-06: "one of them that uses novel payees that are obviously restaurants or utility companies ... category names
# that are similar so instead of just groceries ... groceries and food or eating at home with a grocery card emoji"): two variations of the
# same households (same purchases, days, amounts, regular-vs-new merchants), set by env:
#   RATIONAL_PAYEES=obvious  each merchant renamed (one name per household and merchant) to an invented name that says what it is
#                            ("Hargrove Street Pizzeria", "Linden Electric Cooperative"): a made-up proper part and a cue for the kind;
#                            every name checked against the training merchant pool and the knowledge / alias pairs (never seen verbatim)
#   RATIONAL_CATS=synonyms   each category renamed to a synonym, sometimes with an emoji ("Eating at home 🥦"), one per household; none is a
#                            category name variant of the training households (realstyle.EVERYDAY / CATCHALL)
PROPER = ["Hargrove", "Linden", "Maplewood", "Ashford", "Brightwater", "Calloway", "Delmont", "Eastgate", "Fairhaven", "Glenwood",
          "Hollister", "Ironwood", "Juniper", "Kingsley", "Lakeview", "Millbrook", "Northfield", "Oakhurst", "Pemberton", "Quarry Hill",
          "Riverside", "Stonebridge", "Thornton", "Union Square", "Valley Forge", "Westbrook", "Yardley", "Zephyr", "Birchwood", "Cedar Point",
          "Dunmore", "Elmhurst", "Foxcroft", "Greystone", "Harbor View", "Kessler", "Larkspur", "Marlowe", "Nettleton", "Old Mill"]
OBVIOUS = {
    "grocery": ["{p} Family Grocers", "{p} Fresh Market", "{p} Supermarket", "{p} Food Mart", "{p} Organic Grocery"],
    "restaurant": ["{p} Street Pizzeria", "{p} Italian Kitchen", "{p} Sushi House", "{p} Thai Restaurant", "{p} Steakhouse", "{p} Taqueria"],
    "fast_food": ["{p} Burger Shack", "{p} Fried Chicken", "{p} Taco Stand", "{p} Hot Dogs & Fries"],
    "coffee_bakery": ["{p} Coffee Roasters", "{p} Bakery", "{p} Espresso Bar", "{p} Donut Shop", "{p} Bagel Bakery"],
    "bar": ["{p} Tavern", "{p} Brewing Company", "{p} Wine Bar", "{p} Pub & Taproom"],
    "gas": ["{p} Gas Station", "{p} Fuel Stop", "{p} Gas & Go", "{p} Petroleum"],
    "rent": ["{p} Home Mortgage", "{p} Mortgage Servicing", "{p} Home Loans"],
    "utility": ["{p} Electric Cooperative", "{p} Water Utility", "{p} Power & Light", "{p} Natural Gas Company"],
    "phone": ["{p} Wireless", "{p} Mobile Phone Service", "{p} Cellular", "{p} Telecom & Internet"],
    "pharmacy": ["{p} Pharmacy", "{p} Drug Store", "{p} Rx & Pharmacy"],
    "medical": ["{p} Family Medicine", "{p} Dental Clinic", "{p} Urgent Care", "{p} Pediatrics", "{p} Eye Doctor"],
    "big_box": ["{p} Discount Store", "{p} General Store", "{p} Variety Store", "{p} Department Store"],
    "clothing": ["{p} Clothing Co", "{p} Apparel", "{p} Shoe Store", "{p} Outfitters"],
    "electronics": ["{p} Electronics", "{p} Computer Store", "{p} Phone & Tablet Repair"],
    "home_improvement": ["{p} Hardware", "{p} Lumber & Supply", "{p} Home Center", "{p} Paint Store"],
    "furniture": ["{p} Furniture", "{p} Mattress Store", "{p} Home Furnishings"],
    "car_repair": ["{p} Auto Repair", "{p} Tire & Brake", "{p} Collision Center", "{p} Oil Change"],
    "parking": ["{p} Parking Garage", "{p} Parking Lot", "City of {p} Parking Meters"],
    "entertainment": ["{p} Cinema", "{p} Bowling Lanes", "{p} Mini Golf", "{p} Movie Theater"],
    "hobby": ["{p} Craft Supplies", "{p} Bookstore", "{p} Music Shop", "{p} Hobby & Games"],
    "events": ["{p} Concert Tickets", "{p} Theater Box Office", "{p} Fairgrounds Tickets"],
    "pet": ["{p} Pet Supply", "{p} Animal Hospital", "{p} Veterinary Clinic", "{p} Dog Grooming"],
}
SYNONYMS = {
    "Groceries": ["Groceries & food 🛒", "Eating at home 🥦", "Food shopping", "Supermarket run 🛒", "Kitchen staples"],
    "Restaurants": ["Meals out 🍽️", "Dining & takeaway", "Restaurants & cafes ☕", "Dinners out", "Takeaway 🥡"],
    "Gas": ["Fuel for the car ⛽", "Filling up", "Petrol ⛽", "Gasoline"],
    "Mortgage": ["House payment 🏠", "Home loan", "Our house", "Monthly housing"],
    "Utilities": ["Bills & utilities 💡", "Power, water & phone", "Household bills", "Lights & internet"],
    "Health": ["Doctor & pharmacy 💊", "Medical stuff", "Wellness 🩺", "Prescriptions & copays"],
    "Shopping": ["Stuff for the house 🛍️", "General shopping", "Household goods", "Things we buy"],
    "Car": ["Car upkeep 🚗", "Auto & parking", "Vehicle costs", "Keeping the car running 🔧"],
    "Fun": ["Fun & hobbies 🎉", "Weekend fun 🎳", "Leisure", "Good times"],
    "Pets": ["Dog & cat 🐾", "Pet supplies", "Fur babies 🐶", "Animal care"],
}


def _seen_names():
    """lower-cased merchant names training could have seen: the training merchant pool and the knowledge / alias pairs (name, text)"""
    from ai_experiments.paths import ROOT
    seen = {m["name"].lower() for m in json.loads((PROCESSED / "realstyle_merchants_v1.json").read_text())["merchants"] if m["split"] == "train"}
    for f in ("knowledge_pairs_v1.jsonl", "alias_pairs_v1.jsonl"):
        p = ROOT / "data" / "interim" / f
        if p.exists():
            for line in open(p):
                r = json.loads(line)
                for k in ("name", "text", "kind_text"):
                    if isinstance(r.get(k), str):
                        seen.add(r[k].lower())
    return seen


def _check_synonyms():
    import re
    from ai_experiments import realstyle as R
    train = {n.lower() for _, ns, *_ in R.EVERYDAY for n in ns} | {c.lower() for c in R.CATCHALL}
    bad = [v for vs in SYNONYMS.values() for v in vs if re.sub(r"[^\w&,' ]", "", v).strip().lower() in train]
    assert not bad, bad


def _pool():
    by = {}
    for m in json.loads((PROCESSED / "realstyle_merchants_v1.json").read_text())["merchants"]:
        if m["split"] == "test":
            by.setdefault(m["kind"], []).append(m["name"])
    return by


def budgets(level="clean", n=N):
    pool, out = _pool(), []
    payees_mode, cats_mode = os.environ.get("RATIONAL_PAYEES", ""), os.environ.get("RATIONAL_CATS", "")
    seen = _seen_names() if payees_mode == "obvious" else set()
    if cats_mode == "synonyms":
        _check_synonyms()
    for h in range(n):
        rng = random.Random(220_000 + h)
        reg = {(c, k): rng.sample(pool[k], r) for c, (ks, _, r) in CATS.items() for k in ks}
        r2 = random.Random(234_000 + h)  # row 234's draws: never the household's own rng, so purchases and amounts stay the same
        r3 = random.Random(234_500 + h)  # the synonyms own stream: obvious payees are the same with or without them
        shown = {c: r3.choice(SYNONYMS[c]) if cats_mode == "synonyms" else c for c in CATS}
        cats = [dict(id=f"c{i}", name=shown[c], category_group_id="g", hidden=False, deleted=False) for i, c in enumerate(CATS)]
        alias, used = {}, set()

        def rename(name, k):
            if name not in alias:
                for _ in range(1000):
                    cand = r2.choice(OBVIOUS[k]).format(p=r2.choice(PROPER))
                    if cand.lower() not in seen and cand not in used:
                        break
                alias[name] = cand
                used.add(cand)
            return alias[name]
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
            if payees_mode == "obvious":  # the original string is still drawn (the household's rng advances as before), then replaced
                text = rename(name, k) if level == "clean" else statements.render_v2(rename(name, k), r2)
            pid = payees.setdefault(text, f"p{len(payees)}")
            lo, hi = AMT[c]
            amt = fixed.setdefault(name, round(rng.uniform(lo, hi), 2)) if c == "Mortgage" else round(lo * (hi / lo) ** rng.random(), 2)
            txs.append(dict(id=f"r-{h}-{i:05d}", date=d.isoformat(), amount=-round(amt * 1000), category_id=cid[c], payee_id=pid, approved=True,
                            deleted=False, reason=["known" if name in met else "new"],  # scoring segments only, never model input
                            kind=k))  # the merchant's kind: model input only through li_decider's KINDLINE (a simulated lookup)
            met.add(name)
        tag = ("-obv" if payees_mode == "obvious" else "") + ("-syn" if cats_mode == "synonyms" else "")
        out.append(dict(id=f"rational-{level}{tag}-{h}", category_groups=[dict(id="g", name="Spending")], categories=cats,
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
