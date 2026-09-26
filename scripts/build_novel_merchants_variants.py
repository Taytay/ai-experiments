"""Rendering variants of the novel-merchant item set (PLAN step 62, owner's request 2026-09-26): the same 1,198 items as
novel_merchants_v1 (same merchants, users, shots, options and gold), with only the query's statement string changed, so the three
sets pair item by item and isolate what the rendering costs.

  v1        (frozen)  the generator's renderings: bank prefixes, names cut to 8 or 10 characters or abbreviated ("SNOWFOX*")
  v1_full             the full name in capitals with bank noise kept: a processor prefix, a store number, city and state
                      ("POS DEBIT SNOWFOX SUSHI #4821 MARYSVILLE CA")
  v1_clean            the clean name first, as an enrichment service shows it ("Snowfox Sushi, Marysville CA")
City and state come from Overture by each item's place id. The record lines (`prompt_ctx`) are rebuilt the same way.
usage: uv run --with duckdb python scripts/build_novel_merchants_variants.py [--force]
"""
import json
import os
import random
import sys
from pathlib import Path

from ai_experiments import real6 as R6
from ai_experiments.paths import PROCESSED

SRC = PROCESSED / "novel_merchants_v1.json"
PLACES = Path(os.environ.get("OVERTURE_PLACES", Path.home() / "projects/YNAB/data/overture/places/2026-09-23.1"))
PREFIXES = ["POS DEBIT ", "CARD PURCHASE ", "CHECKCARD {d} ", "DEBIT CARD PURCHASE ", "SQ *", "TST* ", "", ""]


def places(ids):
    import duckdb
    con = duckdb.connect(); con.sql("set threads=16")
    con.sql("create table want (id varchar)"); con.executemany("insert into want values (?)", [[i] for i in ids])
    rows = con.sql(f"""select p.id, p.addresses[1].locality, p.addresses[1].region from read_parquet('{PLACES}/*.parquet') p
                       join want w on p.id = w.id""").fetchall()
    return {i: (loc, reg.split("-")[-1]) for i, loc, reg in rows}


def full(name, loc, st, rng):
    pre = rng.choice(PREFIXES).format(d=f"{rng.randint(1, 12):02d}/{rng.randint(1, 28):02d}")
    num = f" #{rng.randint(1, 9999):04d}" if rng.random() < 0.5 else ""
    return f"{pre}{name.upper().replace('&', 'AND')}{num} {loc.upper()} {st}"


def clean(name, loc, st):
    return f"{name}, {loc.title() if loc.isupper() else loc} {st}"


def swap(it, text):
    q_old = f"Transaction: {it['text']} | ${it['amount']:.2f} | {it['weekday']}\nCategory:"
    q_new = f"Transaction: {text} | ${it['amount']:.2f} | {it['weekday']}\nCategory:"
    assert it["prompt"].endswith(q_old) and it["prompt_ctx"].endswith(q_old), it["id"]
    return dict(it, text=text, prompt=it["prompt"][:-len(q_old)] + q_new, prompt_ctx=it["prompt_ctx"][:-len(q_old)] + q_new)


if __name__ == "__main__":
    src = json.loads(SRC.read_text()); items = src["items"]
    loc = places([it["overture_id"] for it in items])
    assert len(loc) == len(items), (len(loc), len(items))
    for variant in ("full", "clean"):
        out = PROCESSED / f"novel_merchants_v1_{variant}.json"
        if out.exists() and "--force" not in sys.argv:
            sys.exit(f"{out} exists (frozen); pass --force to rebuild")
        rng = random.Random(6201)
        new = [swap(it, full(it["merchant"], *loc[it["overture_id"]], rng) if variant == "full" else clean(it["merchant"], *loc[it["overture_id"]])) for it in items]
        doc = dict(src, name=f"novel_merchants_{variant}", variant=variant, derived_from=dict(file=SRC.name, sha256=src["sha256"]), items=new, sha256=R6.sha256(new))
        out.write_text(json.dumps(doc, indent=0, ensure_ascii=False))
        print(variant, len(new), "items; e.g.", [x["text"] for x in new[:3]])
