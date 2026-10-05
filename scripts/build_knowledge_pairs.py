"""PLAN row 212 (c), knowledge pretraining for the history encoder: (bank string, kind text) pairs for a contrastive stage of the bge-small
encoder before its history training, so world knowledge about merchants and places sits in its weights. One row per pair: `text` is the
merchant or place as a bank would print it (statements.render_v2, the measured statement grammar, as build_knowledge_episodes.py renders
its merchants), `kind_text` is its taxonomy_v2 kind in words ("gas station"), `detail` the source's own words for it where it has any
(Wikidata / Curlie description, Overture's category and its parent: "beauty supply store (personal care and beauty store)").

Sources:
  merchant DB  ~/projects/YNAB/data/merchants/merchants_v1.db (build_merchant_db.py): the TOP most popular merchants of its "train" split,
               the selection and order of build_knowledge_episodes.merchants (the merchant_knowledge_v1_* sets: 20k or 48k), REPS fresh
               renderings each (a domain stem or domain 40% of the time, as there). Licences per source: Wikidata CC0-1.0, Curlie and
               DMOZ CC-BY-3.0 (attribution "Curlie.org" / "DMOZ, the Open Directory Project"), Overture brands (its places' licences).
  Overture     ~/projects/YNAB/data/overture/places/2026-09-23.1 (PROVENANCE.md beside it), US places, open, confidence >= OV_CONF, every
               source's licence CDLA-Permissive-2.0, Apache-2.0 or CC0-1.0; each row keeps the place's id and sources (dataset, licence,
               record id). Two parts: unbranded places from a OV_PCT% system sample of the files, one per distinct name, OV_N of them;
               and every brand's places from the whole of the files, at most BRAND_CAP per brand (REPORT §167 Scaling merchant knowledge:
               coverage beats repetition, so many brands a few times each rather than the big chains many times). Kind by
               taxonomy_v2.overture_kind(basic_category); Overture's vague categories (merchant_db_match_real.VAGUE, §168 Kind lines from
               a lookup: its mapping errors) and non-merchant kinds are dropped.
Left out: the merchant DB's held-out ("test") names, so merchant_knowledge_v2_test's held-out tier stays held out; novel_merchants_v1 places
(by id and name); blind_v1 / blind_v2 merchant names.
Writes data/interim/knowledge_pairs_<VERSION>.jsonl (gitignored; shuffled) and knowledge_pairs_<VERSION>.manifest.json (counts by origin,
kind and licence, the settings, attribution). CPU only; DuckDB memory capped by DUCKDB_MEM.
env: TOP (48000), REPS (2), OV_N (100000), OV_PCT (2), OV_FILES (0 = all 16), OV_CONF (0.8), BRAND_CAP (3), BRAND_MIN (2: places a brand
     needs), LIMIT (0 = all; else a random LIMIT of the pairs), VERSION (v1), DUCKDB_MEM (3GB), SEED (212).
usage: uv run --with duckdb python scripts/build_knowledge_pairs.py
       TOP=1000 OV_FILES=1 OV_PCT=0.5 OV_N=1000 LIMIT=2000 VERSION=smoke uv run --with duckdb python scripts/build_knowledge_pairs.py
"""
import json
import os
import random
import re
import sqlite3
from collections import Counter
from pathlib import Path

from ai_experiments import taxonomy_v2 as T
from ai_experiments.paths import PROCESSED, ROOT
from ai_experiments.statements import render_v2

DATA = Path.home() / "projects" / "YNAB" / "data"
DB = DATA / "merchants" / "merchants_v1.db"
PLACES = DATA / "overture" / "places" / "2026-09-23.1"
TOP, REPS = int(os.environ.get("TOP", "48000")), int(os.environ.get("REPS", "2"))
OV_N, OV_PCT, OV_FILES = int(os.environ.get("OV_N", "100000")), float(os.environ.get("OV_PCT", "2")), int(os.environ.get("OV_FILES", "0"))
OV_CONF, BRAND_CAP, BRAND_MIN = float(os.environ.get("OV_CONF", "0.8")), int(os.environ.get("BRAND_CAP", "3")), int(os.environ.get("BRAND_MIN", "2"))
LIMIT, VERSION, SEED = int(os.environ.get("LIMIT", "0")), os.environ.get("VERSION", "v1"), int(os.environ.get("SEED", "212"))
DUCKDB_MEM = os.environ.get("DUCKDB_MEM", "3GB")
OPEN = ("CDLA-Permissive-2.0", "Apache-2.0", "CC0-1.0")
NOT_MERCHANT = {"several", "purpose", "p2p", "income", "savings", "loan"}  # build_knowledge_episodes.KIND_NAMES leaves these out
KIND_NAMES = {k: v for k, v in T.KINDS.items() if k not in NOT_MERCHANT}
VAGUE = {"media_service", "professional_service", "shopping", "social_or_community_service", "community_and_government"}  # merchant_db_match_real.VAGUE
DB_LICENCE = {"wikidata": "CC0-1.0", "curlie": "CC-BY-3.0", "dmoz": "CC-BY-3.0", "overture": "Overture places (CDLA-Permissive-2.0 / Apache-2.0 / CC0-1.0)"}
ATTRIBUTION = {
    "CC-BY-3.0": "Curlie rows: \"Curlie.org\", CC BY 3.0; DMOZ rows: \"DMOZ, the Open Directory Project\" (final dump, March 2017), CC BY 3.0",
    "Overture": "Data from Meta, Microsoft, PinMeTo, Krick, RenderSEO, DAC and BrightQuery, available under CDLA Permissive 2.0; Copyright 2024 "
                "Foursquare Labs, Inc., available under Apache 2.0; data from AllThePlaces, available under CC0 1.0 (exact text: "
                f"{PLACES / 'PROVENANCE.md'})",
}


def key(s):
    return re.sub(r"[^a-z0-9]", "", (s or "").lower())


def excluded():
    """Normalised names (and Overture ids) that evaluation sets hold out."""
    names, ids = set(), set()
    for f in ("novel_merchants_v1.json", "blind_v1.json", "blind_v2.json"):
        for it in json.loads((PROCESSED / f).read_text())["items"]:
            names.add(key(it.get("merchant"))); ids.add(it.get("overture_id"))
    con = sqlite3.connect(DB)
    names |= {key(n) for (n,) in con.execute("SELECT name FROM merchants WHERE split = 'test'")}
    names.discard(""); ids.discard(None)
    return names, ids


def merchants(n):
    """The n most popular "train" merchants, one per name: build_knowledge_episodes.merchants' query and order (that script reads its
    environment at import, so the query is repeated here), with each row's source and source id kept."""
    con = sqlite3.connect(DB)
    rows = con.execute("""SELECT name, domain, kind, description, path, source, source_id FROM merchants WHERE split = 'train' AND kind IN ({})
                          ORDER BY CASE WHEN rank IS NOT NULL THEN rank WHEN us_places IS NOT NULL THEN 200000 - us_places * 10
                                        WHEN source = 'wikidata' THEN 300000 ELSE 900000 END""".format(",".join("?" * len(KIND_NAMES))),
                       tuple(KIND_NAMES)).fetchall()
    out, seen = [], set()
    for name, dom, kind, desc, path, src, sid in rows:
        k = key(name)
        if k and k not in seen and len(name) <= 60:
            seen.add(k); out.append(dict(name=name, domain=dom, kind=kind, description=desc, path=path, source=src, source_id=sid))
        if len(out) >= n:
            break
    shared = Counter(m["domain"] for m in out if m["domain"])
    for m in out:  # a domain several merchants share is a host (facebook.com, sourceforge.net), not the merchant: render the name only
        if shared[m["domain"]] > 1:
            m["domain"] = ""
    return out


def bank_string(m, rng):
    """build_knowledge_episodes.bank_string (shared host domains aside, see merchants()): the name, or (an online merchant, 40% of the time) its domain stem or domain, rendered."""
    base = m["name"]
    if m["domain"] and rng.random() < 0.4:
        base = m["domain"].split(".")[0] if rng.random() < 0.5 else m["domain"]
    return render_v2(base, rng)


def words(cat):
    return (cat or "").replace("_", " ")


def overture(rng, skip_names, skip_ids):
    import duckdb
    files = sorted(str(p) for p in PLACES.glob("*.parquet"))
    files = files[:OV_FILES] if OV_FILES else files
    con = duckdb.connect()
    con.execute(f"SET memory_limit='{DUCKDB_MEM}'; SET threads=4; SET preserve_insertion_order=false")
    lic = ", ".join(f"'{x}'" for x in OPEN)
    cols = ", ".join(f"{n} := {e}" for n, e in [
        ("id", "id"), ("name", 'names."primary"'), ("brand", 'brand.names."primary"'), ("basic_category", "basic_category"),
        ("tprimary", 'taxonomy."primary"'), ("hierarchy", "taxonomy.hierarchy"), ("city", "addresses[1].locality"),
        ("region", "addresses[1].region"),
        ("sources", "list_transform(sources, s -> struct_pack(dataset := s.dataset, license := s.license, record_id := s.record_id))")])
    where = f"""addresses[1].country = 'US' AND operating_status IS DISTINCT FROM 'permanently_closed' AND confidence >= {OV_CONF}
                AND names."primary" IS NOT NULL AND basic_category IS NOT NULL AND len(sources) > 0
                AND len(list_filter(sources, s -> s.license IS NULL OR s.license NOT IN ({lic}))) = 0"""
    src = f"read_parquet({files!r})"
    plain = con.execute(f"""SELECT arg_min(p, hash(p.id)) FROM (SELECT struct_pack({cols}) p
                            FROM (SELECT * FROM {src} TABLESAMPLE {OV_PCT}% (system, {SEED})) WHERE {where} AND brand.names."primary" IS NULL)
                            GROUP BY lower(p.name)""").fetchall()
    brands = con.execute(f"""SELECT arg_min(p, hash(p.id), {BRAND_CAP}), count(*) FROM (SELECT struct_pack({cols}) p
                             FROM {src} WHERE {where} AND brand.names."primary" IS NOT NULL) GROUP BY lower(p.brand) HAVING count(*) >= {BRAND_MIN}""").fetchall()
    out, drop = [], Counter()

    def keep(p, part, n=None):
        kind = T.overture_kind(p["basic_category"])
        if p["basic_category"] in VAGUE or kind not in KIND_NAMES:
            drop["vague or not a merchant kind"] += 1; return
        if p["id"] in skip_ids or key(p["name"]) in skip_names or key(p["brand"]) in skip_names:
            drop["held out"] += 1; return
        h = [x for x in (p["hierarchy"] or []) if x]
        parent = h[-2] if len(h) >= 2 and h[-2] != p["tprimary"] else None
        city = f"{p['city']} {(p['region'] or '').split('-')[-1]}".upper() if p["city"] else None
        out.append(dict(text=render_v2(p["name"].strip(), rng, city=city), kind=kind, kind_text=KIND_NAMES[kind],
                        detail=words(p["tprimary"]) + (f" ({words(parent)})" if parent else ""), origin=part, name=p["name"].strip(),
                        brand=p["brand"], brand_places=n, overture_id=p["id"], overture_category=p["basic_category"],
                        licences=sorted({s["license"] for s in p["sources"]}), sources=p["sources"]))

    rng.shuffle(plain)
    for (p,) in plain:
        if len(out) >= OV_N:
            break
        keep(p, "overture_place")
    for ps, n in brands:
        for p in ps:
            keep(p, "overture_brand", n)
    return out, dict(files=len(files), sampled_unbranded_names=len(plain), brands=len(brands), dropped=dict(drop))


if __name__ == "__main__":
    rng = random.Random(SEED)
    skip_names, skip_ids = excluded()
    pairs, ndb_skip = [], 0
    for m in merchants(TOP):
        if key(m["name"]) in skip_names:
            ndb_skip += 1; continue
        detail = (m["description"] or m["path"] or "").strip()[:120]
        lic = DB_LICENCE[m["source"]]
        for _ in range(REPS):
            pairs.append(dict(text=bank_string(m, rng), kind=m["kind"], kind_text=KIND_NAMES[m["kind"]], detail=detail,
                              origin=f"merchant_db:{m['source']}", name=m["name"], source_id=m["source_id"], licences=[lic]))
    print(f"merchant DB: {len(pairs) // max(REPS, 1)} merchants x {REPS} renderings ({ndb_skip} held-out names skipped)", flush=True)
    ov, ov_info = overture(rng, skip_names, skip_ids)
    print(f"Overture: {len(ov)} places ({ov_info})", flush=True)
    pairs += ov
    rng.shuffle(pairs)
    if LIMIT:
        pairs = pairs[:LIMIT]
    out = ROOT / "data" / "interim" / f"knowledge_pairs_{VERSION}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        for i, p in enumerate(pairs):
            f.write(json.dumps(dict(id=f"KP{i}", **p), ensure_ascii=False) + "\n")
    origin = Counter(p["origin"] for p in pairs)
    manifest = dict(
        name=f"knowledge_pairs_{VERSION}", plan_row=212, arm="c", file=str(out.relative_to(ROOT)), n_pairs=len(pairs),
        distinct_names=len({key(p["name"]) for p in pairs}), by_origin=dict(origin.most_common()),
        by_kind=dict(Counter(p["kind_text"] for p in pairs).most_common()),
        licences=dict(Counter(lic for p in pairs for lic in p["licences"]).most_common()),
        overture_source_datasets=dict(Counter(f"{s['dataset']} ({s['license']})" for p in pairs for s in p.get("sources", [])).most_common()),
        overture=ov_info, attribution=ATTRIBUTION,
        settings=dict(TOP=TOP, REPS=REPS, OV_N=OV_N, OV_PCT=OV_PCT, OV_FILES=OV_FILES, OV_CONF=OV_CONF, BRAND_CAP=BRAND_CAP, BRAND_MIN=BRAND_MIN,
                      LIMIT=LIMIT, SEED=SEED, overture_release="2026-09-23.1", merchant_db=str(DB)))
    mp = out.with_suffix(".manifest.json")
    mp.write_text(json.dumps(manifest, indent=1, ensure_ascii=False))
    print(f"{len(pairs)} pairs -> {out} ({out.stat().st_size / 1e6:.1f} MB); by origin {dict(origin)}; licences {manifest['licences']}; -> {mp}")
