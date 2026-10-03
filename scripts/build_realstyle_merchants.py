"""PLAN step 173: the merchant pool of the real-style household generator (ai_experiments.realstyle). Real US businesses from Overture
places (release 2026-09-23.1; CDLA-Permissive-2.0 / Apache-2.0 / CC0 sources, each place keeps its id and sources), grouped by
taxonomy_v2 kind: for every kind, the largest brands (one row per brand, by US place count) and a reservoir sample of independents;
plus the MCC list's airline, car-rental and hotel brands (codes 3000-3999, public domain). Each merchant gets a split by a hash of its
id or brand name: "train" (training episodes) or "test" (the real-style test set), so a test household's merchants are unseen in
training (the facts-versus-skill read of REPORT 73 again).
usage: uv run --with duckdb python scripts/build_realstyle_merchants.py [--force]
"""
import csv
import hashlib
import json
import sys
from collections import Counter

from ai_experiments import taxonomy_v2 as T
from ai_experiments.paths import PROCESSED, ROOT

PLACES = "/home/taytay/projects/YNAB/data/overture/places/2026-09-23.1"
OUT = PROCESSED / "realstyle_merchants_v1.json"
BRANDS_PER_KIND, INDEP_PER_KIND, SEED = 150, 1200, 173


def split_of(key):
    return "test" if int(hashlib.sha256(key.lower().encode()).hexdigest(), 16) % 4 == 0 else "train"  # a quarter held out


if __name__ == "__main__":
    import duckdb
    if OUT.exists() and "--force" not in sys.argv:
        sys.exit(f"{OUT} exists (frozen); pass --force to rebuild")
    con = duckdb.connect(); con.sql("SET threads=16; SET memory_limit='20GB'")
    cats = [r[0] for r in con.execute(f"SELECT DISTINCT basic_category FROM read_parquet('{PLACES}/*.parquet') WHERE basic_category IS NOT NULL").fetchall()]
    kind_of = {c: T.overture_kind(c) for c in cats if T.overture_kind(c)}
    con.execute("CREATE TEMP TABLE km(basic VARCHAR, kind VARCHAR)"); con.executemany("INSERT INTO km VALUES (?, ?)", list(kind_of.items()))
    con.sql(f"""CREATE TEMP TABLE cand AS
        SELECT p.id, p.names."primary" AS name, p.brand.names."primary" AS brand, km.kind, p.basic_category AS basic,
               p.addresses[1].locality AS city, p.addresses[1].region AS region,
               list_transform(p.sources, s -> struct_pack(dataset := s.dataset, license := s.license, record_id := s.record_id)) AS sources
        FROM read_parquet('{PLACES}/*.parquet') p JOIN km ON p.basic_category = km.basic
        WHERE p.addresses[1].country = 'US' AND p.operating_status IS DISTINCT FROM 'permanently_closed' AND p.confidence >= 0.7
          AND p.names."primary" IS NOT NULL AND p.addresses[1].locality IS NOT NULL AND p.addresses[1].region IS NOT NULL""")
    out = []
    brand_rows = con.execute(f"""SELECT kind, brand, count(*) n, any_value(id), any_value(basic), any_value(sources) FROM cand WHERE brand IS NOT NULL
                                 GROUP BY kind, brand QUALIFY row_number() OVER (PARTITION BY kind ORDER BY count(*) DESC) <= {BRANDS_PER_KIND}""").fetchall()
    for kind, brand, n, pid, basic, sources in brand_rows:
        out.append(dict(name=brand, kind=kind, basic=basic, chain=True, places=n, city=None, overture_id=pid, sources=sources, split=split_of(brand)))
    for kind in sorted(set(kind_of.values())):
        rows = con.execute(f"""SELECT id, name, basic, city, region, sources FROM (SELECT * FROM cand WHERE kind = ? AND brand IS NULL ORDER BY id)
                               USING SAMPLE reservoir({INDEP_PER_KIND} ROWS) REPEATABLE ({SEED})""", [kind]).fetchall()
        for pid, name, basic, city, region, sources in rows:
            out.append(dict(name=name, kind=kind, basic=basic, chain=False, places=1, city=f"{city.upper()} {region.upper()}", overture_id=pid,
                            sources=sources, split=split_of(pid)))
    mcc = csv.DictReader(open(ROOT / "data" / "external" / "mcc_codes" / "mcc_codes.csv", encoding="utf-8"))
    for r in mcc:
        k = T.mcc_kind(r["mcc"])
        if 3000 <= int(r["mcc"]) <= 3999 and k:
            nm = r["edited_description"].split("(")[0].strip().title()
            out.append(dict(name=nm, kind=k, basic=None, chain=True, places=None, city=None, mcc=r["mcc"], sources=[{"dataset": "mcc-codes", "license": "Unlicense"}],
                            split=split_of(nm)))
    doc = dict(name="realstyle_merchants", version=1, seed=SEED, overture_release="2026-09-23.1", n=len(out),
               note="Overture places by taxonomy_v2 kind (top brands + independents) and MCC airline / car-rental / hotel brands; split by hash",
               merchants=out)
    OUT.write_text(json.dumps(doc) + "\n")
    c = Counter((m["kind"], m["split"]) for m in out)
    print(f"{len(out)} merchants; kinds {len({m['kind'] for m in out})}; test share {sum(m['split'] == 'test' for m in out) / len(out):.2f}; "
          f"size {OUT.stat().st_size / 1e6:.1f} MB")
    print(sorted(Counter(m["kind"] for m in out).items(), key=lambda x: -x[1])[:60])
