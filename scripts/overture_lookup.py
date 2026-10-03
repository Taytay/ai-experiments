"""PLAN step 172 (a) (owner, 2026-10-03: "Could we use our large, free POI database to learn about merchants? ... a lookup that matches
an input payee to it, in case we might be able to inject a payee description at inference time"). Match each payee of the owner's
budget to Overture places and write a kind for it, for real_budget_eval.py's KIND=1 prompt line.

Tables (built once with DuckDB from ~/projects/YNAB/data/overture/places/2026-09-23.1, written beside it in derived/, public data only):
  brands.parquet    brand name -> places, most common basic_category (worldwide, brands with >= 2 places)
  us_names.parquet  lower-cased US place name -> places, most common basic_category
Matching (REPORT 159's tools): payee strings cleaned as payee_resolution_real.clean (the YNAB payee name and, when it has them, the
payee's raw import strings); candidates blocked on the first cleaned word; Jaro-Winkler on the cleaned strings; a brand match first
(score >= BRAND_MIN), else a US place name (score >= NAME_MIN, preferring names with more places); a person-to-person or transfer string is
never matched (raw strings included); every word of the payee must match a word of the place (covered()). The kind is Overture's basic_category made readable ("coffee_shop" -> "coffee shop").
Writes OUT/overture_kinds.json ({payee name: {kind, source, score, match}}), private like everything under OUT; prints aggregates only.
env: BUDGET, OUT (as real_budget_eval.py), BRAND_MIN (0.93), NAME_MIN (0.95).
usage: BUDGET=<id> uv run --with rapidfuzz --with duckdb python scripts/overture_lookup.py
"""
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from payee_resolution_real import NOISE, clean  # noqa: E402

BUDGET = os.environ["BUDGET"]
CACHE = Path.home() / ".cache" / "ynab-cli" / f"{BUDGET}.json"
OUT = Path(os.environ.get("OUT", Path.home() / ".local" / "share" / "ynab-real-eval" / BUDGET))
DERIVED = Path.home() / "projects" / "YNAB" / "data" / "overture" / "derived"
BRAND_MIN = float(os.environ.get("BRAND_MIN", "0.93"))
NAME_MIN = float(os.environ.get("NAME_MIN", "0.95"))
P2P = re.compile(r"\b(paid|transfer|zelle|venmo|cash app|check|deposit|interest|atm)\b", re.I)


def covered(q, cand, truncated):
    """Every word of the cleaned payee (3+ letters) matches a word of the candidate (Jaro-Winkler >= 0.9, or a prefix of it); the last
    word may be unmatched when the original string looks truncated. Stops Jaro-Winkler's shared-prefix bias ("progressive ins" ->
    "progressive roofing")."""
    from rapidfuzz.distance import JaroWinkler
    qs = [w for w in q.split() if len(w) >= 3 and w not in NOISE]
    cs = cand.split()
    miss = [w for w in qs if not any(c.startswith(w) or JaroWinkler.similarity(w, c) >= 0.9 for c in cs)]
    return not miss or (truncated and miss == qs[-1:] and len(qs) > 1)


def first(c):
    w = [x for x in c.split() if x not in NOISE and len(x) > 1]
    return w[0] if w else ""


if __name__ == "__main__":
    import duckdb
    from rapidfuzz.distance import JaroWinkler

    b = json.loads(CACHE.read_text())["budget"]
    names = {p["id"]: p["name"] for p in b["payees"]}
    raw = defaultdict(set)
    for t in b["transactions"]:
        if t.get("import_payee_name_original") and t.get("payee_id"):
            raw[names.get(t["payee_id"], "")].add(t["import_payee_name_original"])
    items = json.loads((OUT / "items.json").read_text())["items"]
    payees = sorted({it["prompt"].rsplit("Transaction: ", 1)[1].split(" | ")[1] for it in items})
    queries = {p: [(c, len(o) >= 18) for o, c in dict.fromkeys((o, clean(o)) for o in [p] + sorted(raw.get(p, ()))[:5]) if c and not P2P.search(o)]
               for p in payees}
    words = sorted({first(c) for cs in queries.values() for c, _ in cs if len(first(c)) >= 3})

    con = duckdb.connect()
    con.execute("CREATE TEMP TABLE w(word VARCHAR)"); con.executemany("INSERT INTO w VALUES (?)", [(x,) for x in words])
    def load(table, col):
        rows = con.execute(f"""SELECT {col}, n, basic_category FROM '{DERIVED}/{table}.parquet'
                               WHERE lower(split_part(regexp_replace(lower({col}), '[^a-z ]', ' ', 'g'), ' ', 1)) IN (SELECT word FROM w)
                                  OR lower(split_part(trim(regexp_replace(lower({col}), '^(the) ', '', 'g')), ' ', 1)) IN (SELECT word FROM w)""").fetchall()
        block = defaultdict(list)
        for name, n, cat in rows:
            c = clean(name)
            if c:
                block[first(c)].append((c, n, cat, name))
        return block
    brands, places = load("brands", "brand"), load("us_names", "name")
    print(f"{len(payees)} payees, {len(words)} first words; candidate brands {sum(map(len, brands.values()))}, US place names {sum(map(len, places.values()))}")

    out = {}
    for p, cs in queries.items():
        if P2P.search(p) or not cs:
            continue
        best = None
        for source, block, th in (("brand", brands, BRAND_MIN), ("place", places, NAME_MIN)):
            for c, trunc in cs:
                for cand, n, cat, orig in block.get(first(c), ()):
                    s = JaroWinkler.similarity(c, cand)
                    one_word = len([w for w in c.split() if w not in NOISE]) < 2
                    if one_word and (source != "brand" or n < 20 or s < 0.99):  # one generic word ("dollar", "made"): a big brand, exact, only
                        continue
                    if s >= th and (best is None or (s, n) > (best[0], best[1])) and covered(c, cand, trunc):
                        best = (s, n, cat, orig, source)
            if best:
                break
        if best:
            out[p] = dict(kind=best[2].replace("_", " "), source=best[4], score=round(best[0], 3), match=best[3], places=best[1])
    dest = OUT / "overture_kinds.json"
    dest.write_text(json.dumps(out)); os.chmod(dest, 0o600)
    n_tx = len(items)
    hit = [it for it in items if it["prompt"].rsplit("Transaction: ", 1)[1].split(" | ")[1] in out]
    new = [it for it in items if not it["payee_seen"]]
    print(f"payees matched: {len(out)} of {len(payees)} ({sum(v['source'] == 'brand' for v in out.values())} by brand); "
          f"transactions covered {100 * len(hit) / n_tx:.1f}%, first-time-payee transactions covered "
          f"{100 * sum(1 for it in new if it in hit) / len(new):.1f}%")
