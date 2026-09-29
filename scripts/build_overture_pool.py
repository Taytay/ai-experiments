"""A training pool of real US businesses from Overture places (PLAN step 107, MODEL-17 / REAL-19): the novel-merchant builder's query and
category mapping (`build_novel_merchants.query`, MAP, `group`) with another seed, every place of the novel-merchant test set
(novel_merchants_v1, by Overture id and by name) and every blind_v1 merchant name left out. exp_categoriser.py's OVDB episodes file
these places under the training user's category for their standard category, so a reader learns what real business names mean,
the knowledge the encoders lack on new local businesses (REPORT 103). Each place keeps its Overture id and per-row sources (licences).
usage: uv run --with duckdb python scripts/build_overture_pool.py [--force]
"""
import json
import sys
from pathlib import Path

from ai_experiments import real6 as R6
from ai_experiments import transactions as T
from ai_experiments.paths import PROCESSED

sys.path.insert(0, str(Path(__file__).parent))
import build_novel_merchants as B  # noqa: E402

OUT = PROCESSED / "overture_pool_v1.json"

if __name__ == "__main__":
    if OUT.exists() and "--force" not in sys.argv:
        sys.exit(f"{OUT} exists (frozen); pass --force to rebuild")
    B.SEED = 107  # another reservoir sample than the test set's 62
    test = json.loads((PROCESSED / "novel_merchants_v1.json").read_text())["items"]
    blind = json.loads((PROCESSED / "blind_v1.json").read_text())["items"]
    ids = {i["overture_id"] for i in test}
    names = {i["merchant"].lower() for i in test} | {i["merchant"].lower() for i in blind} | {m["name"].lower() for m in T.load()["merchants"]}
    pool, seen = [], set()
    for p in B.query():
        n = p["name"].strip()
        if p["id"] in ids or n.lower() in names or n.lower() in seen or len(n) > 40:
            continue
        g = B.group(p)
        if g is None:
            continue
        seen.add(n.lower())
        pool.append(dict(name=n, std=p["std"], group=g, city=f"{p['city']} {p['region'].split('-')[-1]}".upper(), overture_id=p["id"],
                         overture_category=p["tprimary"], sources=p["sources"]))
    doc = dict(name="overture_pool", version="v1", seed=B.SEED, source="Overture Maps places 2026-09-23.1 (per-row licences in `sources`)",
               excluded="novel_merchants_v1 places (id and name), blind_v1 merchant names, the REAL-6 merchants", places=pool, sha256=R6.sha256(pool))
    OUT.write_text(json.dumps(doc, indent=0, ensure_ascii=False))
    from collections import Counter
    print(len(pool), "places;", Counter(p["group"] for p in pool), Counter(p["std"] for p in pool))
