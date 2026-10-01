"""The injected places database for PLAN step 66 (REAL-21, facts or skill): half of POI-1's test places, chosen by a hash of the Overture
id, as database rows (text, name, Overture basic category, amount), for exp_categoriser.py's POI database episodes (POI_DB, DBEP). The
other half is held out: scored on the same test items, the in-database half reads what the weights store about those places (facts),
the held-out half what the episodes teach about places in general (skill). No test item's place is in any user's history (POI-1's
construction), and every place keeps its Overture id and sources.
usage: uv run python scripts/build_poi1_db.py [--force]
"""
import hashlib
import json
import sys

from ai_experiments import real6 as R6
from ai_experiments.paths import PROCESSED

SRC, DST = PROCESSED / "poi1_v1.json", PROCESSED / "poi1_v1_db.json"


def in_db(overture_id):
    return int(hashlib.sha256(overture_id.encode()).hexdigest(), 16) % 2 == 0


if __name__ == "__main__":
    if DST.exists() and "--force" not in sys.argv:
        sys.exit(f"{DST} exists (frozen); pass --force to rebuild")
    doc = json.loads(SRC.read_text())
    places = {}
    for it in doc["items"]:
        places.setdefault(it["overture_id"], dict(overture_id=it["overture_id"], text=it["text"], merchant=it["merchant"], basic=it["basic"],
                                                  amount=it["amount"], sources=it["sources"]))
    db = [p for oid, p in sorted(places.items()) if in_db(oid)]
    out = dict(version="v1_db", source=f"poi1_v1 ({doc['sha256'][:12]})", rule="sha256(overture_id) % 2 == 0",
               places=db, in_db_items=sorted(it["id"] for it in doc["items"] if in_db(it["overture_id"])), sha256=R6.sha256(db))
    DST.write_text(json.dumps(out, indent=0, ensure_ascii=False))
    print(f"{len(places)} test places, {len(db)} in the database; {len(out['in_db_items'])} of {len(doc['items'])} test items in it")
