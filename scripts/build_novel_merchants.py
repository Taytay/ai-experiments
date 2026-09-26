"""Build and freeze the novel-merchant item set (PLAN step 62, QUESTIONS.md REAL-19) from Overture Maps places.

Real US businesses that are in no training database and no REAL-6 user's history, each put to one REAL-6 user as a transaction to
file in that user's own scheme, with that user's frozen 24 shots. Three groups per standard category:
  descriptive  independent businesses whose name contains a word for their kind ("Harbor Pet Supply")
  plain        independent businesses whose name contains no category word of any of the twelve ("Marlowe & Finch")
  chain        places whose Overture brand has at least CHAIN_MIN US locations among the candidates (the model may know it)
Places with a brand below CHAIN_MIN count as independent. Overture's own category is used as the label, so its errors are kept
("Cash Wise Liquor" is a grocery_store there): the set measures real-database behaviour, label noise included.
The category comes from Overture's taxonomy (`taxonomy.primary`, or `basic_category` for whole groups) mapped to REAL-6's twelve
standard categories; ambiguous types (furniture, convenience stores, car dealers, bars) are left out. Each item's `record` is a
real-database record built from Overture's category ("Harbor Pet Supply is listed as a pet store."), for the record-in-prompt arms.
Every item keeps the place's Overture `id`, confidence and `sources` (dataset, licence, record id): the licence is per row
(data/external/overture_places_2026-09-23.1/PROVENANCE.md).

Frozen as data/processed/novel_merchants_v1.json with a sha256 over the items; REAL-6's item format, so exp_real6.py scores it with
ITEMS_SET=novel_merchants_v1 (levels are R6_unseen_<name type of the user's category> so the per-cell summaries work unchanged).
usage: uv run --with duckdb python scripts/build_novel_merchants.py [--force]      (needs the Overture places parquet locally)
"""
import json
import os
import random
import re
import sys
from pathlib import Path

from ai_experiments import merchants as M
from ai_experiments import real6 as R6
from ai_experiments import transactions as T
from ai_experiments.paths import PROCESSED

VERSION = "v1"
OUT = PROCESSED / f"novel_merchants_{VERSION}.json"
PLACES = Path(os.environ.get("OVERTURE_PLACES", Path.home() / "projects/YNAB/data/overture/places/2026-09-23.1"))
PER_CAT = {"descriptive": 40, "plain": 40, "chain": 20}
CHAIN_MIN = 25
SEED = 62
# (column, value) -> standard category; taxonomy.primary is the finer field, basic_category takes whole groups
MAP = {"Groceries": ("primary", ["grocery_store", "butcher_shop", "produce_store", "specialty_foods_store", "health_food_store"]),
       "Restaurants": ("basic", ["restaurant", "casual_eatery", "fast_food_restaurant", "coffee_shop", "cafe"]),
       "Gas & Auto": ("primary", ["gas_station", "automotive_repair", "auto_body_shop", "tire_dealer_and_repair", "car_wash", "tire_shop", "auto_parts_store"]),
       "Clothing": ("primary", ["clothing_store", "womens_clothing_store", "mens_clothing_store", "childrens_clothing_store", "shoe_store", "fashion_boutique"]),
       "Electronics": ("primary", ["electronics_store", "computer_store", "camera_and_photography_store", "audio_visual_equipment_store"]),
       "Home Improvement": ("primary", ["hardware_store", "home_improvement_store", "building_supply_store", "lumber_store", "nursery_and_gardening_store"]),
       "Pharmacy & Health": ("primary", ["pharmacy", "drugstore", "pharmacy_and_drug_store", "dentist", "dental_clinic", "primary_care_or_general_clinic", "optometrist", "vision_or_eye_care_clinic"]),
       "Entertainment": ("primary", ["movie_theater", "bowling_alley", "arcade", "amusement_park", "theatre_venue"]),
       "Travel": ("primary", ["hotel", "motel", "travel_agent"]),
       "Pets": ("primary", ["pet_store", "animal_and_pet_store", "veterinarian", "pet_groomer", "pet_boarding"]),
       "Fitness": ("basic", ["gym", "fitness_studio"]),
       "Telecom & Utilities": ("primary", ["telecommunications_service", "mobile_phone_store", "electric_utility_provider", "water_utility_provider", "natural_gas_utility_provider"])}
# a name is "descriptive" when it contains one of its category's words (whole-word, case-insensitive)
WORDS = {"Groceries": "grocery|groceries|market|supermarket|foods|food mart|meats?|butcher|produce|farm stand",
         "Restaurants": "restaurant|grill|cafe|café|kitchen|diner|pizza|pizzeria|taco|tacos|burger|burgers|bbq|sushi|bistro|eatery|coffee|bakery|deli",
         "Gas & Auto": "auto|automotive|tire|tires|motors?|car wash|gas|fuel|collision|body shop|garage|lube|muffler|brake",
         "Clothing": "boutique|apparel|clothing|clothiers?|shoes?|fashions?|outfitters|wear|threads",
         "Electronics": "electronics|computers?|cameras?|audio|tech|digital|photo",
         "Home Improvement": "hardware|lumber|building supply|home center|garden|nursery|supply|paint",
         "Pharmacy & Health": "pharmacy|pharmacies|drug|drugs|apothecary|dental|dentistry|dentists?|dds|dmd|d\\.d\\.s|orthodontics?|clinics?|medical|eyes?|optical|vision|optometry|family practice|health|pediatrics?|physicians?|doctors?",
         "Entertainment": "cinemas?|theaters?|theatres?|movies?|bowl|bowling|lanes|arcades?|fun|amusement|playhouse|entertainment",
         "Travel": "hotel|inn|motel|suites|lodge|resort|travel|tours?",
         "Pets": "pets?|veterinary|vet|animal|grooming|groomers?|paws?|dog|dogs|cat|kennel",
         "Fitness": "fitness|gym|yoga|pilates|crossfit|training|barre|athletic|strength|boxing|martial",
         "Telecom & Utilities": "wireless|mobile|cellular|telecom|communications?|electric|power|water|utilities|utility|gas company|broadband|phones?|cell"}


def query():
    import duckdb
    con = duckdb.connect(); con.sql("set threads=16")
    case = " ".join(f"when {'taxonomy."primary"' if col == 'primary' else 'basic_category'} in ({', '.join(repr(v) for v in vals)}) then {cat!r}"
                    for cat, (col, vals) in MAP.items())
    con.sql(f"""create table cand as
                select id, names."primary" as name, brand.names."primary" as brand, taxonomy."primary" as tprimary, basic_category, confidence,
                       addresses[1].locality city, addresses[1].region region,
                       list_transform(sources, s -> struct_pack(dataset := s.dataset, license := s.license, record_id := s.record_id)) sources,
                       case {case} end std
                from read_parquet('{PLACES}/*.parquet')
                where addresses[1].country = 'US' and operating_status is distinct from 'permanently_closed' and confidence >= 0.8
                  and names.primary is not null and addresses[1].locality is not null and addresses[1].region is not null""")
    con.sql("delete from cand where std is null")
    con.sql("create table cand2 as select *, count(*) over (partition by lower(brand)) brand_n from cand")
    rows = []
    for cat in MAP:  # USING SAMPLE applies before WHERE, so sample from the per-category subquery
        q = f"select * from (select * from cand2 where std = {cat!r} order by id) using sample reservoir(4000 rows) repeatable ({SEED})"
        cols = [d[0] for d in con.sql(q).description]
        rows += [dict(zip(cols, r)) for r in con.sql(q).fetchall()]
    return rows


def group(p):
    if p["brand"] and p["brand_n"] >= CHAIN_MIN:
        return "chain"
    if re.search(rf"\b({WORDS[p['std']]})\b", p["name"], re.I):
        return "descriptive"
    if any(re.search(rf"\b({w})\b", p["name"], re.I) for w in WORDS.values()):
        return None  # a category word, but another category's: neither clean group
    return "plain"


def build():
    rng = random.Random(SEED)
    doc = R6.load(); users = doc["users"]
    known = {m["name"].lower() for m in T.load()["merchants"]}
    by_user_items = {}
    for it in doc["items"]:
        by_user_items.setdefault(it["user"], it)  # one frozen REAL-6 item per user: its prompt carries the user's 24 shots
    places = query()
    chosen = []
    for cat in MAP:
        pool = [p for p in places if p["std"] == cat and p["name"].lower() not in known and len(p["name"]) <= 40]
        seen = set()
        for g, n in PER_CAT.items():
            cand = [p for p in pool if group(p) == g and p["name"].lower() not in seen]
            if g == "chain":  # one place per brand, so twenty different chains
                byb = {}
                for p in cand:
                    byb.setdefault(p["brand"].lower(), p)
                cand = list(byb.values())
            rng.shuffle(cand)
            for p in cand[:n]:
                seen.add(p["name"].lower()); chosen.append(dict(p, group=g))
    items, k = [], 0
    for p in chosen:
        order = users[:]; rng.shuffle(order)
        for u in order:  # the first user whose scheme names this category unambiguously (not split)
            match = [i for i, c in enumerate(u["categories"]) if "split" not in c and p["std"] in c["standard"]]
            if match:
                break
        else:
            continue
        ci = match[0]; c = u["categories"][ci]
        m = dict(name=p["name"], city=f"{p['city']} {p['region'].split('-')[-1]}".upper())
        text = T.render(m, rng)
        mu, sig = T.AMOUNT[p["std"]]; amount = round(rng.lognormvariate(mu, sig), 2); weekday = rng.choice(T.WEEKDAYS)
        record = f"{p['name']} is listed as a {p['tprimary'].replace('_', ' ')}."
        base = by_user_items[u["user"]]
        head, sep, _ = base["prompt"].rpartition("\nTransaction: ")
        head_ctx = base["prompt_ctx"].rpartition("\nTransaction: ")[0].rpartition("\nNote: ")[0]
        q = f"Transaction: {text} | ${amount:.2f} | {weekday}\nCategory:"
        items.append(dict(id=f"NM:{k:04d}", level=f"R6_unseen_{c['name_type']}", user=u["user"], merchant=p["name"], known=p["group"] == "chain",
                          text=text, amount=amount, weekday=weekday, options=[" " + x["name"] for x in u["categories"]], answer=ci,
                          prompt=head + "\n" + q, prompt_ctx=head_ctx + f"\nNote: {record}\n" + q, record=record, name_type=c["name_type"], seen=False,
                          nm_group=p["group"], std=p["std"], overture_id=p["id"], overture_category=p["tprimary"], overture_confidence=round(p["confidence"], 3),
                          brand=p["brand"], sources=p["sources"]))
        k += 1
    return items


if __name__ == "__main__":
    if OUT.exists() and "--force" not in sys.argv:
        sys.exit(f"{OUT} exists (frozen); pass --force to rebuild")
    items = build()
    doc = dict(name="novel_merchants", version=VERSION, n_items=len(items), source="Overture Maps places 2026-09-23.1 (per-row licences in `sources`; "
               "provenance in data/external/overture_places_2026-09-23.1/)", seed=SEED, per_category=PER_CAT, mapping=MAP, items=items, sha256=R6.sha256(items))
    OUT.write_text(json.dumps(doc, indent=0, ensure_ascii=False))
    from collections import Counter
    print(len(items), "items;", Counter(i["nm_group"] for i in items), Counter(i["std"] for i in items))
    print("licences:", Counter(s["license"] for i in items for s in i["sources"]))
