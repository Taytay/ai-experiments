"""Build and freeze the label-induction item set (PLAN step 64, QUESTIONS.md REAL-20): when can a model infer what a meaningless
category name means from the user's examples in the prompt?

Every item is a scheme of the twelve standard categories, 24 labelled examples and one query, all real US businesses from Overture
(clean renderings, "Name, City ST", so every business's kind is readable). The query's category (gold) carries a freshly generated
word ("Tavoli") that no model has seen in training, so the item measures induction from the prompt alone. One base condition and
factors varied one at a time, each on the same 300 queries (25 per standard category), so conditions pair item by item:

  base          2 gold examples, of another kind of business in the query's standard category (query a pizzeria, examples a
                taqueria and a cafe); 3 coined categories (gold + 2 others) in the scheme; no decoy
  n_gold        0, 1, 2 (base), 4, 8 gold examples
  kind          same kind as the query (same Overture taxonomy.primary) / other kind, same category (base) / opaque names
                (generated, meaningless: "Oskpobury Group")
  n_coined      1, 3 (base), 6 coined categories in the scheme
  decoy         base + one example of the query's own kind filed under another category / the same with no gold examples
  control       the gold category keeps its standard name (the name's meaning available)

The other categories' examples are descriptive businesses of their own category (at least one each, 24 in all); amounts from the
category's distribution. Every item keeps the Overture ids and source licences of its query (sources) and examples (example_ids).
Frozen as data/processed/label_induction_<version>.json in REAL-6's item format (score with ITEMS_SET=label_induction_<version> USERS=all).

v1 turned out to be solvable by elimination (a blind Opus reader: 100% with no gold examples): the scheme lacks exactly one standard
category and has exactly one coined word left unexplained by the examples. v2 (--empty 3) adds three more coined categories with no
examples to every scheme, as users have categories with no recent transactions, so elimination leaves a guess among four words and
only the gold examples can decide; everything else is v1's, item for item.
usage: uv run --with duckdb python scripts/build_label_induction.py [--version v2 --empty 3] [--force]
"""
import json
import random
import sys
from pathlib import Path

from ai_experiments import merchants as M
from ai_experiments import real6 as R6
from ai_experiments import transactions as T
from ai_experiments.paths import PROCESSED

sys.path.insert(0, str(Path(__file__).parent))
import build_novel_merchants as NM  # noqa: E402

VERSION = sys.argv[sys.argv.index("--version") + 1] if "--version" in sys.argv else "v1"
EMPTY = int(sys.argv[sys.argv.index("--empty") + 1]) if "--empty" in sys.argv else 0  # coined categories with no examples (v2: 3)
OUT = PROCESSED / f"label_induction_{VERSION}.json"
SEED, N_QUERY, N_SHOTS = 64, 25, 24
BASE = dict(n_gold=2, kind="other_kind", n_coined=3, decoy=False)
CONDITIONS = [("base", {}), ("n_gold=0", dict(n_gold=0)), ("n_gold=1", dict(n_gold=1)), ("n_gold=4", dict(n_gold=4)), ("n_gold=8", dict(n_gold=8)),
              ("kind=same_kind", dict(kind="same_kind")), ("kind=opaque", dict(kind="opaque")), ("n_coined=1", dict(n_coined=1)), ("n_coined=6", dict(n_coined=6)),
              ("decoy", dict(decoy=True)), ("decoy, n_gold=0", dict(decoy=True, n_gold=0)), ("control: standard name", dict(coined_gold=False))]
SYLL = [c + v for c in "bdfgklmnprstvz" for v in "aeiou"]


def fresh_word(rng, taken):
    while True:
        w = ("".join(rng.choice(SYLL) for _ in range(rng.randint(2, 3))) + rng.choice(["", "n", "r", "x", "l"])).capitalize()
        if w not in taken:
            taken.add(w)
            return w


def clean(p):
    return f"{p['name']}, {p['city'].title() if p['city'].isupper() else p['city']} {p['region'].split('-')[-1]}"


def row(p, cat, rng):
    mu, sig = T.AMOUNT[cat]
    return dict(text=clean(p), amount=round(rng.lognormvariate(mu, sig), 2), weekday=rng.choice(T.WEEKDAYS), id=p["id"])


def opaque_row(cat, rng, taken):
    m = M.build_extra(1, taken=taken, seed=rng.randrange(10**9))[0]; taken.add(m["name"])
    mu, sig = T.AMOUNT[cat]
    return dict(text=f"{m['name']}, {m['city'].split()[0].title()} {m['city'].split()[-1]}", amount=round(rng.lognormvariate(mu, sig), 2), weekday=rng.choice(T.WEEKDAYS), id=None)


def build():
    rng = random.Random(SEED)
    places = [p for p in NM.query() if not (p["brand"] and p["brand_n"] >= NM.CHAIN_MIN)]
    used = {i["merchant"].lower() for i in json.loads((PROCESSED / "novel_merchants_v1.json").read_text())["items"]}  # disjoint from row 62's set
    desc = [p for p in places if NM.group(p) == "descriptive" and p["name"].lower() not in used and len(p["name"]) <= 40]
    by_cat, by_kind = {}, {}
    for p in desc:
        by_cat.setdefault(p["std"], []).append(p); by_kind.setdefault((p["std"], p["tprimary"]), []).append(p)
    queries = []
    for cat in M.CATEGORY_LIST:
        pool = [p for p in by_cat[cat] if len(by_kind[(cat, p["tprimary"])]) >= 9 and sum(q["tprimary"] != p["tprimary"] for q in by_cat[cat]) >= 8]
        queries += [p for p in rng.sample(pool, N_QUERY)]
    taken_words = set(R6.NEW_WORDS) | {c["name"] for u in R6.load()["users"] for c in u["categories"]}
    opaque_taken = {m["name"] for m in T.load()["merchants"]}
    items = []
    for qn, q in enumerate(queries):
        qrng = random.Random(SEED * 1000 + qn)  # per query: the same examples across conditions except what the condition changes
        others = [c for c in M.CATEGORY_LIST if c != q["std"]]
        coined_order = qrng.sample(others, 5)  # the other categories to coin, in a fixed order per query
        words = {c: fresh_word(qrng, taken_words) for c in [q["std"]] + coined_order}
        empties = [f"__empty{k}" for k in range(EMPTY)]  # after the query's other draws, so v1's items are unchanged when EMPTY = 0
        words.update({e: fresh_word(random.Random(SEED * 7919 + qn * 31 + k), taken_words) for k, e in enumerate(empties)})
        same_kind = [p for p in by_kind[(q["std"], q["tprimary"])] if p["id"] != q["id"]]
        other_kind = [p for p in by_cat[q["std"]] if p["tprimary"] != q["tprimary"]]
        gold_pool = {"same_kind": qrng.sample(same_kind, min(8, len(same_kind))), "other_kind": qrng.sample(other_kind, min(8, len(other_kind)))}
        opaque_pool = [opaque_row(q["std"], qrng, opaque_taken) for _ in range(8)]
        other_rows = {c: [row(p, c, qrng) for p in qrng.sample(by_cat[c], 6)] for c in others}
        decoy_row = row(qrng.choice([p for p in same_kind if p not in gold_pool["same_kind"]] or same_kind), q["std"], qrng)
        decoy_cat = qrng.choice(others)
        qrow = row(q, q["std"], qrng)
        for cname, change in CONDITIONS:
            c = {**BASE, "coined_gold": True, **change}
            coined = ([q["std"]] if c["coined_gold"] else []) + coined_order[:c["n_coined"] - 1]  # the control keeps the same other coined categories
            names = {k: (words[k] if k in coined else k) for k in M.CATEGORY_LIST}
            names.update({e: words[e] for e in empties})
            gold = [row(p, q["std"], random.Random(qn * 7 + j)) for j, p in enumerate(gold_pool.get(c["kind"], [])[:c["n_gold"]])] if c["kind"] != "opaque" else opaque_pool[:c["n_gold"]]
            shots = [(r, q["std"]) for r in gold]
            slots = N_SHOTS - len(shots) - (1 if c["decoy"] else 0)
            per = {k: 1 for k in others}
            for j in range(slots - len(others)):  # the rest spread round-robin over the other categories, in a fixed order
                per[others[j % len(others)]] += 1
            for k in others:
                shots += [(r, k) for r in other_rows[k][:per[k]]]
            if c["decoy"]:
                shots.append((decoy_row, decoy_cat))
            random.Random(SEED + qn).shuffle(shots)
            opts = M.CATEGORY_LIST[:]; random.Random(qn).shuffle(opts)
            for k, e in enumerate(empties):  # the empty coined categories at fixed positions per query
                opts.insert(random.Random(qn * 13 + k).randrange(len(opts) + 1), e)
            header = "Categories: " + ", ".join(names[k] for k in opts) + "\n\n"
            demo = "".join(f"Transaction: {r['text']} | ${r['amount']:.2f} | {r['weekday']}\nCategory: {names[k]}\n\n" for r, k in shots)
            query = f"Transaction: {qrow['text']} | ${qrow['amount']:.2f} | {qrow['weekday']}\nCategory:"
            record = f"{q['name']} is listed as a {q['tprimary'].replace('_', ' ')}."
            items.append(dict(id=f"LI:{qn:03d}:{cname}", level="R6_unseen_" + ("new" if c["coined_gold"] else "standard"), user=qn % 20, merchant=q["name"], known=False,
                              text=qrow["text"], amount=qrow["amount"], weekday=qrow["weekday"], options=[" " + names[k] for k in opts], answer=opts.index(q["std"]),
                              prompt=header + demo + query, prompt_ctx=header + demo + f"Note: {record}\n" + query, record=record,
                              name_type="new" if c["coined_gold"] else "standard", seen=False, condition=cname, query=qn, std=q["std"], overture_category=q["tprimary"],
                              overture_id=q["id"], sources=q["sources"], example_ids=[r["id"] for r, _ in shots], factors=c))
    return items


if __name__ == "__main__":
    if OUT.exists() and "--force" not in sys.argv:
        sys.exit(f"{OUT} exists (frozen); pass --force to rebuild")
    items = build()
    doc = dict(name="label_induction", version=VERSION, empty_coined_categories=EMPTY, n_items=len(items), conditions=[c for c, _ in CONDITIONS], base=BASE, seed=SEED,
               source="Overture Maps places 2026-09-23.1 (per-row licences in `sources`; provenance in data/external/overture_places_2026-09-23.1/)", items=items, sha256=R6.sha256(items))
    OUT.write_text(json.dumps(doc, indent=0, ensure_ascii=False))
    from collections import Counter
    print(len(items), "items;", Counter(i["condition"] for i in items))
