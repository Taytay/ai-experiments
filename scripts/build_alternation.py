"""Users who alternate between categories at one merchant (PLAN step 86, row 43's first piece; owner, 2026-09-27): a user has a more
specific category ("Date night", "Game stuff") and uses it for some of a merchant's transactions. alternation_v1, fold 0's users.

Per user and merchant, a new specific category B is added to the user's list (at a random place) next to the category A the user
files the merchant's kind under (groceries for general stores, restaurants for restaurants). Six of the 24 shots are this merchant's
rows, filed by the scenario's rule; four queries follow:

  observable    stores: B when the amount is $40 or more, else A; restaurants: B on Fri / Sat, else A (ceiling 100)
  partial       the same rule, each label flipped with probability 0.2, in the history and the query (ceiling 80)
  unobservable  A or B at random, 60 / 40, whatever the amount and day (ceiling 60: always A)
  control       always A; B is used by two rows of another merchant (ceiling 100)

Amounts: stores a mixture (half around $20, half around $75), restaurants around $30; the rule's two sides are both queried. Each item
carries `rule_pred` (the category the rule gives, the Bayes answer) and `b_idx` / `a_idx`; the ceiling of a cell is the share of items
whose gold equals `rule_pred`. Merchants: 8 general stores and 8 restaurants, real chains not in REAL-6.
usage: uv run python scripts/build_alternation.py [--force]
"""
import json
import math
import random
import sys

from ai_experiments import merchants as M
from ai_experiments import real6 as R6
from ai_experiments import transactions as T
from ai_experiments.paths import PROCESSED

DST = PROCESSED / "alternation_v1.json"
STORES = ["Target", "Walmart", "Amazon", "Kohl's", "Meijer", "Dollar General", "Fred Meyer", "Big Lots"]
RESTAURANTS = ["The Cheesecake Factory", "Maggiano's", "Texas Roadhouse", "Red Lobster", "P.F. Chang's", "Outback Steakhouse", "Applebee's", "Buca di Beppo"]
B_STORE = ["Game stuff", "Cozy time", "Treat yourself", "Craft corner", "Kid stuff", "Hair care"]
B_REST = ["Date night", "Fun food", "Us time", "Special dinners"]
SCEN = ("observable", "partial", "unobservable", "control")


def amount(kind, rng, side=None):
    if kind == "store":
        side = side if side is not None else rng.random() < 0.5
        while True:
            a = math.exp(rng.gauss(4.3, 0.3)) if side else math.exp(rng.gauss(3.0, 0.4))
            if (a >= 40) == side:
                return round(a, 2)
    return round(math.exp(rng.gauss(3.4, 0.4)), 2)


def draw(kind, scen, rng, side=None):
    """(amount, weekday, is_B, rule_pred_is_B) for one transaction; side forces the rule's side (True = B) for queries."""
    if kind == "store":
        a = amount(kind, rng, side); d = rng.choice(T.WEEKDAYS); rule = a >= 40
    else:
        if side is None:
            d = rng.choice(T.WEEKDAYS)
        else:
            d = rng.choice(["Fri", "Sat"] if side else ["Mon", "Tue", "Wed", "Thu", "Sun"])
        a = amount(kind, rng); rule = d in ("Fri", "Sat")
    if scen == "observable":
        b = rule
    elif scen == "partial":
        b = rule if rng.random() >= 0.2 else not rule
    elif scen == "unobservable":
        b = rng.random() < 0.4; rule = False
    else:
        b = False; rule = False
    return a, d, b, rule


def line(m, rng, a, d):
    return f"Transaction: {T.render(m, rng)} | ${a:.2f} | {d}"


if __name__ == "__main__":
    if DST.exists() and "--force" not in sys.argv:
        sys.exit(f"{DST} exists (frozen); pass --force to rebuild")
    doc = R6.load(); items = []
    for u in doc["users"]:
        if u["user"] % 4:
            continue
        cats = u["categories"]
        std = {}
        for c in cats:
            for s in c["standard"]:
                std.setdefault(s, []).append(c)
        prompt0 = next(it for it in doc["items"] if it["user"] == u["user"])["prompt"]
        header, rest = prompt0.split("\n\n", 1)
        shots = rest.split("\n\n")[:-1]
        for kind, names, bnames, s in (("store", STORES, B_STORE, "Groceries"), ("restaurant", RESTAURANTS, B_REST, "Restaurants")):
            owners = std.get(s, [])
            if len(owners) != 1 or "split" in owners[0]:
                continue
            A = owners[0]["name"]
            for name in names:
                for scen in SCEN:
                    r = random.Random(f"alt-{u['user']}-{name}-{scen}")
                    m = dict(name=name, category=s, city=r.choice(M._CITIES))
                    B = r.choice([b for b in bnames if b not in [c["name"] for c in cats]])
                    cat_names = [c["name"] for c in cats]; cat_names.insert(r.randint(0, len(cat_names)), B)
                    hdr = "Categories: " + ", ".join(cat_names)
                    while True:  # six history rows of the merchant, both labels at least twice (control: all A)
                        hist = [draw(kind, scen, r) for _ in range(6)]
                        nb = sum(h[2] for h in hist)
                        if scen == "control" or 2 <= nb <= 4:
                            break
                    demo = list(shots); slots = r.sample(range(len(demo)), 6 + (2 if scen == "control" else 0))
                    for j, (a, d, b, _) in zip(slots, hist):
                        demo[j] = line(m, r, a, d) + f"\nCategory: {B if b else A}"
                    if scen == "control":  # B in use by another merchant of the same kind
                        other = dict(name=r.choice([n for n in names if n != name]), category=s, city=r.choice(M._CITIES))
                        for j in slots[6:]:
                            a, d, _, _ = draw(kind, "observable", r, side=True)
                            demo[j] = line(other, r, a, d) + f"\nCategory: {B}"
                    for q in range(4):
                        a, d, b, rule = draw(kind, scen, r, side=q % 2 == 1)
                        prompt = hdr + "\n\n" + "\n\n".join(demo) + "\n\n" + line(m, r, a, d) + "\nCategory:"
                        gold = cat_names.index(B if b else A)
                        items.append(dict(id=f"ALT_{kind}_{scen}:{u['user']}:{name}:{q}", level=f"ALT_{kind}_{scen}", user=u["user"], merchant=name,
                                          known=True, text=prompt.rsplit("Transaction: ", 1)[1].split(" | ")[0], amount=a, weekday=d, prompt=prompt,
                                          prompt_ctx=prompt, options=[" " + n for n in cat_names], answer=gold, name_type="new" if b else "standard",
                                          seen=True, kind=kind, scen=scen, a_idx=cat_names.index(A), b_idx=cat_names.index(B),
                                          rule_pred=cat_names.index(B if rule else A)))
    out = dict(version="alternation_v1", variant_of=f"real6 v1 ({doc['sha256'][:12]})", items=items, sha256=R6.sha256(items))
    DST.write_text(json.dumps(out, indent=0, ensure_ascii=False))
    from collections import Counter
    import numpy as np
    print(len(items), "items:", dict(Counter(i["level"] for i in items)))
    for lv in sorted({i["level"] for i in items}):
        print(lv, "ceiling", round(100 * np.mean([i["answer"] == i["rule_pred"] for i in items if i["level"] == lv]), 1))
    ex = next(i for i in items if i["level"] == "ALT_restaurant_observable")
    print("\n".join(l for l in ex["prompt"].split("\n") if ex["merchant"].upper()[:6] in l.upper() or l.startswith("Categories"))[:600])
