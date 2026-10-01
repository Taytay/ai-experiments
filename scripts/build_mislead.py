"""Misleading merchant names (PLAN step 83; owner, 2026-09-27): do facts trained into the weights, or the user's own history, beat a name
that points to another category? REAL-6's opaque merchants (Istlock, Elrholm) have no meaning to override.

Merchants (data/processed/mislead_v1.json):
  misleading  48 made-up names that suggest another category than their own, 4 per REAL-6 category ("Tire Barn" is a restaurant,
              "Pixel Depot" a grocery store); `decoy` is the category the name suggests
  neutral     48 opaque twins in REAL-6's style (merchants.build_extra), the same categories, no meaning
  real        well-known brands whose names suggest another category (Kayak, Cricket Wireless, Buffalo Wild Wings; none a REAL-6 merchant); scoring only, never in the DB
Half of the misleading and half of the neutral merchants, alternating within each category, are in the fact DB (`in_db`); training with
MISLEAD=mislead_v1 (exp_categoriser.py) adds those to the database episodes. None is in any REAL-6 user's history.

Items, fold 0's held-out users (REAL-6 format; the user's own header and 24 shots, the query swapped):
  cond "none"  the merchant alone: in-DB merchants test the facts against the name; out-of-DB ones show where the name alone pulls
  cond "hist"  out-of-DB made-up merchants only: two of the 24 shots replaced by this merchant's rows filed under the user's category
Items are made only where the merchant's category maps to one unsplit category of the user; `decoy_idx` is the option of the category the
name suggests (-1 when that is the gold's own category or split), for the share of answers pulled to it.
usage: uv run python scripts/build_mislead.py [--force] [--all]
"""
import json
import math
import random
import sys

from ai_experiments import merchants as M
from ai_experiments import real6 as R6
from ai_experiments import transactions as T
from ai_experiments.paths import PROCESSED

ALL = "--all" in sys.argv  # every user (each fold's model reads its own users), not fold 0 only
DST = PROCESSED / ("mislead_v1_all.json" if ALL else "mislead_v1.json")
MISLEADING = {  # true category: [(name, the category the name suggests)]
    "Restaurants": [("Tire Barn", "Gas & Auto"), ("Iron Temple", "Fitness"), ("Paws Palace", "Pets"), ("Kiddie Kingdom", "Entertainment")],
    "Groceries": [("Chop Shop Co", "Gas & Auto"), ("Pixel Depot", "Electronics"), ("Wanderlust Lane", "Travel"), ("Timber & Nail", "Home Improvement")],
    "Electronics": [("Fresh Harvest", "Groceries"), ("Happy Tails", "Pets"), ("Sole Mates", "Clothing"), ("Gridiron Grill", "Restaurants")],
    "Pets": [("Byte Barn", "Electronics"), ("Velvet Thread", "Clothing"), ("Pump House", "Fitness"), ("Oven & Ale", "Restaurants")],
    "Fitness": [("Sugar Crumb", "Restaurants"), ("Wired Up", "Electronics"), ("Jet Set Lounge", "Travel"), ("Cozy Nest", "Home Improvement")],
    "Home Improvement": [("Silk Road Journeys", "Travel"), ("Kitty Corner", "Pets"), ("Spin City", "Entertainment"), ("Pill Box", "Pharmacy & Health")],
    "Pharmacy & Health": [("Gearhead Garage", "Gas & Auto"), ("Game Vault", "Entertainment"), ("Tailor Made", "Clothing"), ("Butcher Block", "Groceries")],
    "Entertainment": [("Green Grocer Hall", "Groceries"), ("Signal Tower", "Telecom & Utilities"), ("Wellness Well", "Pharmacy & Health"), ("Brake Point", "Gas & Auto")],
    "Travel": [("Pet Pantry", "Pets"), ("Fit Nation", "Fitness"), ("Stitch Lab", "Clothing"), ("Circuit Hub", "Electronics")],
    "Gas & Auto": [("Bloom Apothecary", "Pharmacy & Health"), ("Arcade Alley", "Entertainment"), ("Bistro Blue", "Restaurants"), ("Bark Avenue", "Pets")],
    "Clothing": [("Fuel Stop", "Gas & Auto"), ("Router Room", "Telecom & Utilities"), ("Harvest Basket", "Groceries"), ("Hammer Time", "Home Improvement")],
    "Telecom & Utilities": [("Sweat Box", "Fitness"), ("Trail Mix Pantry", "Groceries"), ("Rx Corner", "Pharmacy & Health"), ("Voyage Co", "Travel")],
}
REAL = [("Banana Republic", "Clothing", "Groceries"), ("Kayak", "Travel", "Fitness"), ("Hopper", "Travel", "Entertainment"), ("Hotwire", "Travel", "Electronics"),
        ("Cricket Wireless", "Telecom & Utilities", "Entertainment"), ("Mint Mobile", "Telecom & Utilities", "Groceries"), ("Boost Mobile", "Telecom & Utilities", "Fitness"),
        ("SoulCycle", "Fitness", "Gas & Auto"), ("Pure Barre", "Fitness", "Restaurants"), ("Fandango", "Entertainment", "Restaurants"),
        ("GoodRx", "Pharmacy & Health", "Electronics"), ("Valvoline", "Gas & Auto", "Home Improvement"), ("Red Robin", "Restaurants", "Pets"),
        ("Five Guys", "Restaurants", "Clothing"), ("Buffalo Wild Wings", "Restaurants", "Pets"), ("Cracker Barrel", "Restaurants", "Groceries")]


def merchant(rng, name, cat, kind, decoy, in_db):
    return dict(name=name, category=cat, kind=kind, decoy=decoy, in_db=in_db, known=kind == "real", products=rng.sample(M.CATEGORIES[cat], 3),
                city=rng.choice(M._CITIES), n=rng.randint(1, 9999), d=f"{rng.randint(1, 12):02d}/{rng.randint(1, 28):02d}", bank_tmpl=rng.randrange(len(M._BANK_TMPL)))


def row(m, rng):
    mu, sig = T.AMOUNT[m["category"]]
    return dict(text=T.render(m, rng), amount=round(math.exp(rng.gauss(mu, sig)), 2), weekday=rng.choice(T.WEEKDAYS))


if __name__ == "__main__":
    if DST.exists() and "--force" not in sys.argv:
        sys.exit(f"{DST} exists (frozen); pass --force to rebuild")
    rng = random.Random(83)
    taken = {m["name"] for m in T.load()["merchants"]}
    ms = []
    for cat, lst in MISLEADING.items():
        assert cat in M.CATEGORIES and all(d in M.CATEGORIES and d != cat for _, d in lst), cat
        for k, (name, decoy) in enumerate(lst):
            assert name not in taken; taken.add(name)
            ms.append(merchant(rng, name, cat, "misleading", decoy, k % 2 == 0))
    extra = M.build_extra(400, taken=taken, seed=83)
    for cat, lst in MISLEADING.items():
        twins = [e["name"] for e in extra if e["category"] == cat][:len(lst)]
        assert len(twins) == len(lst), cat
        for k, name in enumerate(twins):
            taken.add(name); ms.append(merchant(rng, name, cat, "neutral", None, k % 2 == 0))
    for name, cat, decoy in REAL:
        assert name not in taken; ms.append(merchant(rng, name, cat, "real", decoy, False))
    fact_db = {m["name"]: f"{m['name']} is a store that sells {M.prods(m)}." for m in ms if m["in_db"]}

    doc = R6.load(); items = []
    for u in doc["users"]:
        if u["user"] % 4 and not ALL:
            continue
        cats = u["categories"]; names = [c["name"] for c in cats]
        one = {}
        for c in cats:
            for s in c["standard"]:
                one.setdefault(s, []).append(c)
        base = next(it for it in doc["items"] if it["user"] == u["user"])["prompt"]
        header, rest = base.split("\n\n", 1)
        shots = rest.split("\n\n")[:-1]  # "Transaction: ...\nCategory: ..." blocks; the last block is the query
        for m in ms:
            owners = one.get(m["category"], [])
            if len(owners) != 1 or "split" in owners[0]:
                continue
            gold = names.index(owners[0]["name"])
            dec = one.get(m["decoy"], []) if m["decoy"] else []
            decoy_idx = names.index(dec[0]["name"]) if len(dec) == 1 and "split" not in dec[0] and dec[0]["name"] != names[gold] else -1
            conds = ["none"] + (["hist"] if m["kind"] != "real" and not m["in_db"] else [])
            for cond in conds:
                r = random.Random(f"{u['user']}-{m['name']}-{cond}")
                q = row(m, r)
                demo = list(shots)
                if cond == "hist":
                    for j in r.sample(range(len(demo)), 2):
                        h = row(m, r)
                        demo[j] = f"Transaction: {h['text']} | ${h['amount']:.2f} | {h['weekday']}\nCategory: {names[gold]}"
                prompt = header + "\n\n" + "\n\n".join(demo) + f"\n\nTransaction: {q['text']} | ${q['amount']:.2f} | {q['weekday']}\nCategory:"
                items.append(dict(id=f"ML_{m['kind']}_{cond}:{u['user']}:{m['name']}", level=f"ML_{m['kind']}_{cond}", user=u["user"], merchant=m["name"],
                                  known=m["known"], text=q["text"], amount=q["amount"], weekday=q["weekday"], prompt=prompt, prompt_ctx=prompt,
                                  options=[" " + n for n in names], answer=gold, name_type=owners[0]["name_type"], seen=cond == "hist",
                                  mkind=m["kind"], in_db=m["in_db"], cond=cond, decoy=m["decoy"], decoy_idx=decoy_idx))
    out = dict(version="mislead_v1" + ("_all" if ALL else ""), variant_of=f"real6 v1 ({doc['sha256'][:12]})", merchants=ms, fact_db=fact_db, items=items, sha256=R6.sha256(items))
    DST.write_text(json.dumps(out, indent=0, ensure_ascii=False))
    from collections import Counter
    print(len(ms), "merchants,", len(fact_db), "in the DB;", len(items), "items:", dict(Counter(i["level"] for i in items)))
    print("with a decoy option:", sum(i["decoy_idx"] >= 0 for i in items))
    ex = next(i for i in items if i["merchant"] == "Tire Barn")
    print(ex["prompt"][:120].replace("\n", " / "), "...", ex["prompt"][-90:].replace("\n", " / "), "| gold", ex["options"][ex["answer"]])
