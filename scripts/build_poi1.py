"""Build and freeze POI-1 (PLAN step 65, QUESTIONS.md POI-1): user categorisation over real places, at Overture's scale.

200 synthetic users (four folds by user id mod 4, as row 42) over real US places from Overture (release 2026-09-23.1). Each user's
scheme groups 20 to 45 of Overture's basic categories into 12 to 20 of their own: groups follow Overture's top-level taxonomy, split
and merged at random (so some groups cross top-levels, as real users' do), each named with a readable name ("Coffee shop", "Food and
drink"), a merge of two ("Hardware & Garden") or a coined word ("Tavoli"). Each user has a 150-place history (categories weighted
Zipf over the user's own, places never shared between users) and a frozen block of 24 shots stratified over their groups.

Test items (up to 12 per user; 2,053 in v1, as some users have fewer than six unfiled categories) are places in nobody's history, so no merchant lookup answers them: 6 of a basic category the user has
filed before ("seen kind": a new coffee shop), 6 of a basic category in the user's scheme but absent from their history ("unseen
kind": a first bakery, filed where the user files cafes). Rendered clean ("Name, City ST"; the obscure generator rendering is the
`_obscure` variant). REAL-6's item format (levels R6_<seen|unseen>_<standard|renamed|new> where seen = kind seen and the name types are
readable / merged / coined), plus Overture id, sources (per-row licences), basic category, group and chain flag.
usage: uv run --with duckdb python scripts/build_poi1.py [--force]      (needs the Overture places parquet locally)
"""
import json
import math
import os
import random
import sys
from pathlib import Path

from ai_experiments import merchants as M
from ai_experiments import real6 as R6
from ai_experiments import transactions as T
from ai_experiments.paths import PROCESSED

PLACES = Path(os.environ.get("OVERTURE_PLACES", Path.home() / "projects/YNAB/data/overture/places/2026-09-23.1"))
OUT = PROCESSED / "poi1_v1.json"
SEED, N_USERS, HIST, N_SHOTS, TEST_SEEN, TEST_UNSEEN, PER_CAT, MIN_CAT = 65, 200, 150, 24, 6, 6, 600, 2000
TOPS = ["food_and_drink", "shopping", "lifestyle_services", "health_care", "travel_and_transportation", "sports_and_recreation",
        "arts_and_entertainment", "lodging", "services_and_business", "education"]
SYLL = [c + v for c in "bdfgklmnprstvz" for v in "aeiou"]
AMOUNT = {"food_and_drink": (3.0, 0.6), "shopping": (4.0, 0.8), "lifestyle_services": (3.8, 0.6), "health_care": (4.3, 0.9), "travel_and_transportation": (3.9, 0.8),
          "sports_and_recreation": (3.7, 0.7), "arts_and_entertainment": (3.4, 0.7), "lodging": (5.3, 0.7), "services_and_business": (4.6, 1.0), "education": (4.8, 1.0)}


def readable(s):
    return s.replace("_or_", " or ").replace("_and_", " and ").replace("_", " ").capitalize()


def query():
    import duckdb
    con = duckdb.connect(); con.sql("set threads=16")
    tops = ", ".join(repr(t) for t in TOPS)
    con.sql(f"""create table cand as
                select id, names."primary" as name, brand.names."primary" as brand, basic_category as basic, taxonomy.hierarchy[1] as top,
                       addresses[1].locality as city, addresses[1].region as region,
                       list_transform(sources, s -> struct_pack(dataset := s.dataset, license := s.license, record_id := s.record_id)) as sources
                from read_parquet('{PLACES}/*.parquet')
                where addresses[1].country = 'US' and operating_status is distinct from 'permanently_closed' and confidence >= 0.8
                  and names."primary" is not null and length(names."primary") <= 40 and addresses[1].locality is not null
                  and addresses[1].region is not null and basic_category is not null and taxonomy.hierarchy[1] in ({tops})""")
    keep = [b for b, n in con.sql(f"select basic, count(*) from cand where basic not in ({tops}) group by 1 having count(*) >= {MIN_CAT}").fetchall()]
    con.sql("create table cand2 as select *, count(*) over (partition by lower(brand)) brand_n from cand")
    rows = []
    for b in sorted(keep):
        q = f"select * from (select * from cand2 where basic = {b!r} order by id) using sample reservoir({PER_CAT} rows) repeatable ({SEED})"
        cols = [d[0] for d in con.sql(q).description]
        rows += [dict(zip(cols, r)) for r in con.sql(q).fetchall()]
    return rows


def fresh(rng, taken):
    while True:
        w = ("".join(rng.choice(SYLL) for _ in range(rng.randint(2, 3))) + rng.choice(["", "n", "r", "x", "l"])).capitalize()
        if w not in taken:
            taken.add(w)
            return w


def scheme(rng, cats, top_of, weight, taken):
    m = rng.randint(20, 45); g = rng.randint(12, 20)
    chosen = []
    pool = list(cats)
    while len(chosen) < m:
        c = rng.choices(pool, [weight[x] for x in pool])[0]; chosen.append(c); pool.remove(c)
    groups = {}
    for c in chosen:
        groups.setdefault(top_of[c], []).append(c)
    groups = list(groups.values())
    while len(groups) < g:  # split the largest group at random
        big = max(groups, key=len)
        if len(big) < 2:
            break
        groups.remove(big); rng.shuffle(big); k = rng.randint(1, len(big) - 1); groups += [big[:k], big[k:]]
    while len(groups) > g:  # merge the two smallest (they may cross top-levels, as real users' categories do)
        groups.sort(key=len); a, b = groups[0], groups[1]; groups = groups[2:] + [a + b]
    used, out = set(), []
    for grp in groups:
        r = rng.random(); tops = {top_of[c] for c in grp}
        if len(grp) == 1 and r < 0.45:
            name, nt = readable(grp[0]), "standard"
        elif len(tops) == 1 and r < 0.7 and readable(next(iter(tops))) not in used:
            name, nt = readable(next(iter(tops))), "standard"
        elif len(grp) > 1 and r < 0.8:
            a, b = rng.sample(grp, 2); name, nt = f"{readable(a)} & {readable(b).lower()}", "renamed"
        else:
            name, nt = fresh(rng, taken), "new"
        if name in used:
            name, nt = fresh(rng, taken), "new"
        used.add(name); out.append(dict(name=name, name_type=nt, basic=sorted(grp)))
    return out


def clean(p):
    return f"{p['name']}, {p['city'].title() if p['city'].isupper() else p['city']} {p['region'].split('-')[-1]}"


def build():
    rng = random.Random(SEED)
    places = query()
    by_basic = {}
    for p in places:
        by_basic.setdefault(p["basic"], []).append(p)
    for v in by_basic.values():
        rng.shuffle(v)
    top_of = {b: v[0]["top"] for b, v in by_basic.items()}
    weight = {b: math.sqrt(len(v)) for b, v in by_basic.items()}
    taken = set(R6.NEW_WORDS)
    cursor = {b: 0 for b in by_basic}  # places are consumed in order, so no place is in two histories or in a history and a test

    def take(b):
        if cursor[b] >= len(by_basic[b]):
            return None
        p = by_basic[b][cursor[b]]; cursor[b] += 1
        return p

    def txn(p, r):
        mu, sig = AMOUNT[p["top"]]
        return dict(text=clean(p), amount=round(r.lognormvariate(mu, sig), 2), weekday=r.choice(T.WEEKDAYS), merchant=p["name"], id=p["id"], basic=p["basic"])

    users, items = [], []
    for uid in range(N_USERS):
        cats = scheme(rng, list(by_basic), top_of, weight, taken)
        label_of = {b: c["name"] for c in cats for b in c["basic"]}
        basics = [b for c in cats for b in c["basic"]]; rng.shuffle(basics)
        z = {b: 1 / (k + 1) ** 1.1 for k, b in enumerate(basics)}
        history = []
        while len(history) < HIST:
            b = rng.choices(basics, [z[x] for x in basics])[0]; p = take(b)
            if p is None:
                continue
            history.append(dict(txn(p, rng), label=label_of[b]))
        by_lab = {}
        for h in history:
            by_lab.setdefault(h["label"], []).append(h)
        shots = [rng.choice(v) for v in by_lab.values()]
        rest = [h for h in history if h not in shots]; rng.shuffle(rest); shots = (shots + rest)[:N_SHOTS]; rng.shuffle(shots)
        seen_b = sorted({h["basic"] for h in history}); unseen_b = sorted(set(basics) - set(seen_b))
        header = "Categories: " + ", ".join(c["name"] for c in cats) + "\n\n"
        demo = "".join(f"Transaction: {s['text']} | ${s['amount']:.2f} | {s['weekday']}\nCategory: {s['label']}\n\n" for s in shots)
        tests = [(b, True) for b in rng.sample(seen_b, min(TEST_SEEN, len(seen_b)))] + [(b, False) for b in rng.sample(unseen_b, min(TEST_UNSEEN, len(unseen_b)))]
        for b, seen in tests:
            p = take(b)
            if p is None:
                continue
            t = txn(p, rng); gi = next(k for k, c in enumerate(cats) if b in c["basic"]); c = cats[gi]
            q = f"Transaction: {t['text']} | ${t['amount']:.2f} | {t['weekday']}\nCategory:"
            record = f"{p['name']} is listed as a {readable(b).lower()}."
            items.append(dict(id=f"POI:{uid:03d}:{len(items):05d}", level=f"R6_{'seen' if seen else 'unseen'}_{c['name_type']}", user=uid, merchant=p["name"],
                              known=bool(p["brand"] and p["brand_n"] >= 25), text=t["text"], amount=t["amount"], weekday=t["weekday"],
                              options=[" " + x["name"] for x in cats], answer=gi, prompt=header + demo + q, prompt_ctx=header + demo + f"Note: {record}\n" + q,
                              record=record, name_type=c["name_type"], seen=seen, basic=b, top=p["top"], group=c["name"], group_size=len(c["basic"]),
                              chain=bool(p["brand"] and p["brand_n"] >= 25), overture_id=p["id"], sources=p["sources"]))
        users.append(dict(user=uid, categories=cats, n_categories=len(cats), history=[dict(h, merchant=h["merchant"]) for h in history],
                          shots=[s["text"] for s in shots]))
    return users, items


if __name__ == "__main__":
    if OUT.exists() and "--force" not in sys.argv:
        sys.exit(f"{OUT} exists (frozen); pass --force to rebuild")
    users, items = build()
    doc = dict(name="poi1", version="v1", n_users=len(users), n_items=len(items), seed=SEED, users=users, items=items, sha256=R6.sha256(items),
               source="Overture Maps places 2026-09-23.1 (per-row licences in `sources`; provenance in data/external/overture_places_2026-09-23.1/)")
    OUT.write_text(json.dumps(doc, indent=0, ensure_ascii=False))
    from collections import Counter
    print(len(users), "users,", len(items), "items;", Counter(i["level"] for i in items))
    print("categories per user:", Counter(u["n_categories"] for u in users).most_common(4), "; chains among items:", sum(i["chain"] for i in items))
