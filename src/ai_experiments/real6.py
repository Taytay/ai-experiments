"""REAL-6: an evaluation set shaped like the production task (PLAN step 35).

Synthetic users over the frozen transaction history of `transactions` (240 merchants: 120 real chains, 120 opaque). Each user
has their own category scheme (8 to 20 categories: the 12 standard ones merged or split, each with a standard name, a renamed
one, or a coined word), an imbalanced labelled history of transactions (string, amount, weekday, the user's label), and test
transactions in six cells: the merchant is in the history (seen) or not (unseen; every merchant has a fact-DB record), and its
user category carries a standard, a renamed or a new name. The LLM items are 24-shot prompts of the user's own history with
the user's category names as options (`prompt`), plus a variant with the merchant's fact-DB record before the query
(`prompt_ctx`, the retrieval oracle of row 33). Encoders score the same items by prototype distance from the history.

Frozen once as data/processed/real6_v1.json with a sha256 over the items, like the ladder (items.py). `build()` regenerates.
"""
import hashlib
import json
import random
from collections import Counter, defaultdict

from . import merchants as M
from . import transactions as T
from .paths import PROCESSED

VERSION = "v1"
PATH = PROCESSED / f"real6_{VERSION}.json"
N_USERS, SHOTS, TEST_PER_CELL = 20, 24, 12
RENAMES = {"Groceries": ["Food shopping", "Supermarket", "Grocery run"], "Restaurants": ["Eating out", "Dining", "Food & drink"],
           "Gas & Auto": ["Car", "Fuel and car", "Vehicle"], "Clothing": ["Apparel", "Wardrobe", "Clothes"], "Electronics": ["Gadgets", "Tech", "Devices"],
           "Home Improvement": ["House projects", "Hardware", "Home repairs"], "Pharmacy & Health": ["Medical", "Health", "Drugstore"],
           "Entertainment": ["Fun", "Leisure", "Going out"], "Travel": ["Trips", "Vacation", "Flights and hotels"], "Pets": ["Animals", "Dog and cat", "Pet care"],
           "Fitness": ["Gym", "Exercise", "Workout"], "Telecom & Utilities": ["Bills", "Phone and internet", "Utilities"]}
SPLITS = {"Restaurants": ["Coffee and snacks", "Sit-down meals"], "Groceries": ["Weekly shop", "Top-up shop"], "Travel": ["Work trips", "Holidays"],
          "Entertainment": ["Nights out", "Subscriptions"], "Home Improvement": ["Big projects", "Small fixes"], "Electronics": ["Computers", "Gadgets and media"],
          "Clothing": ["Everyday clothes", "Special occasions"], "Pharmacy & Health": ["Prescriptions", "Wellness"]}
NEW_WORDS = ["Zorbit", "Plenk", "Vandle", "Quorra", "Mistle", "Grendo", "Tabbin", "Oskel", "Yurra", "Blint", "Dravik", "Fennow", "Halmer", "Juvix",
             "Korrel", "Lumbry", "Nastel", "Pravin", "Rondle", "Sibbet", "Tremmo", "Ulvane", "Wexmor", "Zindle"]


def sha256(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")).hexdigest()


def fact_db(merchants, seed=5):
    """One record per merchant: section 4's sentence for the opaque ones, a category-pool sentence for the real chains."""
    rng = random.Random(seed)
    db = {}
    for m in merchants:
        prods = m.get("products") or rng.sample(M.CATEGORIES[m["category"]], 3)
        db[m["name"]] = f"{m['name']} is a store that sells {prods[0]}, {prods[1]} and {prods[2]}."
    return db


AMB_PATH = PROCESSED / f"real6_{VERSION}_ambdb.json"


def fact_db_ambiguous(merchants, seed=5, overlap=2, multi_frac=0.2):
    """REAL-7: the same 240 merchants with section 35's product ambiguity (merchants.build_v2), so a record no longer names its
    category by construction. Each category's pool gains `overlap` products of the next category; a `multi_frac` share of every
    category's merchants (real chains and opaque alike) sell two products of their own category and one of another; the rest draw
    three from the overlapping pool. Returns (name -> record, name -> {multi, secondary, products})."""
    rng = random.Random(seed)
    pools = {}
    for i, c in enumerate(M.CATEGORY_LIST):
        pools[c] = list(M.CATEGORIES[c]) + M.CATEGORIES[M.CATEGORY_LIST[(i + 1) % len(M.CATEGORY_LIST)]][:overlap]
    db, meta = {}, {}
    for c in M.CATEGORY_LIST:
        ms = sorted((m["name"] for m in merchants if m["category"] == c)); rng.shuffle(ms)
        n_multi = round(len(ms) * multi_frac)
        for j, name in enumerate(ms):
            if j < n_multi:
                other = rng.choice([x for x in M.CATEGORY_LIST if x != c])
                prods = rng.sample(M.CATEGORIES[c], 2) + [rng.choice(M.CATEGORIES[other])]; rng.shuffle(prods)
            else:
                other, prods = None, rng.sample(pools[c], 3)
            db[name] = f"{name} is a store that sells {prods[0]}, {prods[1]} and {prods[2]}."
            meta[name] = dict(multi=j < n_multi, secondary=other, products=prods, off_pool=sum(p not in M.CATEGORIES[c] for p in prods))
    return db, meta


def set_record(item, record):
    """The item with another merchant record in its context prompt (the ambiguous DB, or a retrieved record, right or wrong)."""
    head, sep, q = item["prompt_ctx"].rpartition("\nTransaction: ")
    assert head.endswith(f"Note: {item['record']}"), item["id"]
    head = head[:-len(item["record"])] + record
    return dict(item, record=record, prompt_ctx=head + sep + q)


def freeze_amb(force=False):
    if AMB_PATH.exists() and not force:
        raise SystemExit(f"{AMB_PATH.name} exists; frozen sets are immutable.")
    db, meta = fact_db_ambiguous(T.load()["merchants"])
    doc = dict(name="real6_ambdb", version=VERSION, n_merchants=len(db), fact_db=db, meta=meta, sha256=sha256(db))
    AMB_PATH.write_text(json.dumps(doc, indent=0, ensure_ascii=False) + "\n", encoding="utf-8")
    return doc


def make_scheme(rng, merchants):
    """A user's categories: the 12 standard ones merged (fewer) or split (more), each with a name type; returns
    (categories: [{name, name_type, standard: [..]}], merchant -> category index)."""
    n_cat = rng.randint(8, 20)
    cats = [dict(standard=[c]) for c in M.CATEGORY_LIST]
    if n_cat < 12:
        for _ in range(12 - n_cat):  # merge two random categories
            i, j = rng.sample(range(len(cats)), 2)
            cats[i]["standard"] += cats[j]["standard"]; cats.pop(j)
    split_of = {}
    if n_cat > 12:
        pool = [c for c in SPLITS if all(len(x["standard"]) == 1 for x in cats if x["standard"][0] == c)]
        for c in rng.sample(pool, min(n_cat - 12, len(pool))):
            idx = next(i for i, x in enumerate(cats) if x["standard"] == [c])
            cats[idx] = dict(standard=[c], split=0); cats.append(dict(standard=[c], split=1)); split_of[c] = True
    for c in cats:
        if "split" in c:
            c["name"], c["name_type"] = SPLITS[c["standard"][0]][c["split"]], "renamed"
        elif len(c["standard"]) > 1:
            r = rng.random()
            c["name"], c["name_type"] = (" & ".join(RENAMES[s][0] for s in c["standard"]), "renamed") if r < 0.6 else (rng.choice(NEW_WORDS), "new")
        else:
            r = rng.random()
            if r < 0.4:
                c["name"], c["name_type"] = c["standard"][0], "standard"
            elif r < 0.75:
                c["name"], c["name_type"] = rng.choice(RENAMES[c["standard"][0]]), "renamed"
            else:
                c["name"], c["name_type"] = rng.choice(NEW_WORDS), "new"
    names = [c["name"] for c in cats]
    while len(set(names)) < len(names):  # coined words must be distinct within a user
        for i, c in enumerate(cats):
            if names.count(c["name"]) > 1 and c["name_type"] == "new":
                c["name"] = rng.choice([w for w in NEW_WORDS if w not in names]); names = [x["name"] for x in cats]
    assign = {}
    for m in merchants:
        idx = [i for i, c in enumerate(cats) if m["category"] in c["standard"]]
        assign[m["name"]] = idx[0] if len(idx) == 1 else rng.choice(idx)  # a split category: the merchant lands on one side, arbitrarily
    return cats, assign


def build(seed=0, n_users=N_USERS):
    doc = T.load()
    merchants, tx = doc["merchants"], doc["transactions"]
    by_merchant = defaultdict(list)
    for t in tx:
        by_merchant[t["merchant"]].append(t)
    db = fact_db(merchants)
    rng = random.Random(seed)
    users, items = [], []
    for u in range(n_users):
        world = rng.sample(merchants, rng.randint(90, 150))
        cats, assign = make_scheme(rng, world)
        world = [m for m in world if by_merchant[m["name"]]]  # merchants with at least one transaction
        rng.shuffle(world)
        n_hist = int(len(world) * 0.65)
        hist_m, unseen_m = world[:n_hist], world[n_hist:]
        history, test_seen = [], []
        for m in hist_m:
            rows = list(by_merchant[m["name"]]); rng.shuffle(rows)
            keep = max(1, len(rows) - 1) if len(rows) > 1 else 1
            history += rows[:keep]; test_seen += rows[keep:] if len(rows) > 1 else []
        rng.shuffle(history); history = history[:300]
        hist_recs = [dict(text=t["text"], amount=t["amount"], weekday=t["weekday"], merchant=t["merchant"], label=cats[assign[t["merchant"]]]["name"]) for t in history]
        # the 24-shot prompt: stratified over the user's categories, then filled by frequency
        by_cat = defaultdict(list)
        for h in hist_recs:
            by_cat[h["label"]].append(h)
        shots = []
        for c in cats:
            if by_cat[c["name"]]:
                shots.append(rng.choice(by_cat[c["name"]]))
        rest = [h for h in hist_recs if h not in shots]; rng.shuffle(rest)
        shots += rest[:max(0, SHOTS - len(shots))]; rng.shuffle(shots)
        header = "Categories: " + ", ".join(c["name"] for c in cats) + "\n\n"
        demo = "".join(f"Transaction: {h['text']} | ${h['amount']:.2f} | {h['weekday']}\nCategory: {h['label']}\n\n" for h in shots)
        options = [" " + c["name"] for c in cats]
        test_unseen = [t for m in unseen_m for t in by_merchant[m["name"]]]
        cells = defaultdict(list)
        for seen, rows in (("seen", test_seen), ("unseen", test_unseen)):
            rng.shuffle(rows)
            for t in rows:
                c = cats[assign[t["merchant"]]]
                cells[(seen, c["name_type"])].append(t)
        for (seen, nt), rows in sorted(cells.items()):
            for t in rows[:TEST_PER_CELL]:
                c = assign[t["merchant"]]
                q = f"Transaction: {t['text']} | ${t['amount']:.2f} | {t['weekday']}\nCategory:"
                items.append(dict(level=f"R6_{seen}_{nt}", user=u, merchant=t["merchant"], known=t["known"], text=t["text"], amount=t["amount"], weekday=t["weekday"],
                                  prompt=header + demo + q, prompt_ctx=header + demo + f"Note: {db[t['merchant']]}\n" + q, options=options, answer=c,
                                  record=db[t["merchant"]], name_type=nt, seen=seen == "seen"))
        users.append(dict(user=u, categories=cats, n_categories=len(cats), n_history=len(hist_recs), history=hist_recs, shots=[h["text"] for h in shots],
                          history_merchants=[m["name"] for m in hist_m], unseen_merchants=[m["name"] for m in unseen_m]))
    seen_ctr = Counter()
    for it in items:
        it["id"] = f"{it['level']}:{seen_ctr[it['level']]:03d}"; seen_ctr[it["level"]] += 1
    return dict(name="real6", version=VERSION, n_users=len(users), n_items=len(items), shots=SHOTS, users=users, fact_db=db, items=items, sha256=sha256(items))


def freeze(force=False):
    if PATH.exists() and not force:
        raise SystemExit(f"{PATH.name} exists; frozen sets are immutable. Bump VERSION for a new set.")
    doc = build()
    PATH.write_text(json.dumps(doc, indent=0, ensure_ascii=False) + "\n", encoding="utf-8")
    return doc


def load(db="v1"):
    """The frozen set; db="amb" swaps every record (item["record"], item["prompt_ctx"], doc["fact_db"]) for the ambiguous DB of
    `real6_v1_ambdb.json` and records its sha as doc["db_sha256"]; the items' own sha is unchanged (same ids, strings, options, gold)."""
    doc = json.loads(PATH.read_text(encoding="utf-8"))
    assert sha256(doc["items"]) == doc["sha256"], f"{PATH.name}: items do not match the recorded sha256"
    doc["db"] = db
    if db == "amb":
        amb = json.loads(AMB_PATH.read_text(encoding="utf-8"))
        assert sha256(amb["fact_db"]) == amb["sha256"], f"{AMB_PATH.name}: records do not match the recorded sha256"
        doc["items"] = [set_record(it, amb["fact_db"][it["merchant"]]) for it in doc["items"]]
        doc["fact_db"], doc["db_meta"], doc["db_sha256"] = amb["fact_db"], amb["meta"], amb["sha256"]
    elif db != "v1":
        raise ValueError(db)
    return doc


if __name__ == "__main__":
    import sys
    if "--amb" in sys.argv:  # uv run python -m ai_experiments.real6 --amb   freezes the ambiguous DB beside the set
        amb = freeze_amb(force="--force" in sys.argv) if not AMB_PATH.exists() or "--force" in sys.argv else json.loads(AMB_PATH.read_text())
        print(f"ambiguous DB: {amb['n_merchants']} records, sha {amb['sha256'][:12]}, multi {sum(v['multi'] for v in amb['meta'].values())}, "
              f"records with an off-pool product {sum(v['off_pool'] > 0 for v in amb['meta'].values())}")
        raise SystemExit
    doc = freeze(force="--force" in sys.argv) if not PATH.exists() or "--force" in sys.argv else load()
    print(f"{doc['n_users']} users, {doc['n_items']} items, sha {doc['sha256'][:12]}")
    print("cells", Counter(i["level"] for i in doc["items"]))
    print("categories per user", [u["n_categories"] for u in doc["users"]], "| history sizes", [u["n_history"] for u in doc["users"]])
    it = doc["items"][0]; print(it["prompt"][-400:]); print(it["options"], it["answer"])


def db_only_merchants(seed=9, frac=0.25):
    """REAL-5's control (PLAN step 33): a quarter of the merchants, stratified over the standard categories, that NO user's training
    rows may carry. A merchant unseen by one user is usually in another user's history, so a categoriser trained across users learns
    its category from them; the DB-only merchants are the ones whose category can only come from the fact database (or the model's
    pretraining, for the real chains). The frozen set is unchanged; training scripts drop these merchants' rows and the tables split
    the unseen cells by them."""
    ms = T.load()["merchants"]
    rng = random.Random(seed)
    out = set()
    for c in M.CATEGORY_LIST:
        pool = sorted(m["name"] for m in ms if m["category"] == c)
        out |= set(rng.sample(pool, max(1, round(len(pool) * frac))))
    return out
