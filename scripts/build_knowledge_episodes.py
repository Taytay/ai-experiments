"""PLAN step 179: merchant-knowledge episodes from the local merchant database (build_merchant_db.py), packed so one training sequence
teaches many merchants (REPORTs 58 / 73: facts need ~30 exposures each). Each episode is a categorisation prompt in the recipe's format
whose categories are merchant kinds (taxonomy_v2) and whose history rows are merchants rendered as bank strings (statements.render_v2:
names, domains, processor prefixes, codes, truncation), each filed under its kind; the shot-label loss trains every row's label, the
query is one more merchant. Merchants: the TOP most popular of the "train" split (Majestic rank, US chain size; Wikidata entries count as
popular), each appearing EXPOSURES times across episodes; a test set of held-out merchants ("test" split) and of trained ones, as
single-merchant items, measures facts against skill.
Writes data/processed/merchant_knowledge_v1_train.jsonl (episodes like realstyle's: context, options, answer, spans) and
data/processed/merchant_knowledge_v1_test.json (items: id, options, answer, prompt, merchant, trained).
env: TOP (20000), EXPOSURES (30), ROWS (24).
usage: uv run python scripts/build_knowledge_episodes.py
"""
import datetime as dt
import json
import os
import random
import re
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from build_realstyle import label_spans  # noqa: E402

from ai_experiments import taxonomy_v2 as T  # noqa: E402
from ai_experiments.paths import PROCESSED  # noqa: E402
from ai_experiments.statements import render_v2  # noqa: E402

DB = Path.home() / "projects" / "YNAB" / "data" / "merchants" / "merchants_v1.db"
TOP, EXPOSURES, ROWS = int(os.environ.get("TOP", "20000")), int(os.environ.get("EXPOSURES", "30")), int(os.environ.get("ROWS", "24"))
WD = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
KIND_NAMES = {k: v for k, v in T.KINDS.items() if k not in ("several", "purpose", "p2p", "income", "savings", "loan")}


def merchants(split, n):
    """The n most popular merchants of one split, one row per name (best-ranked source wins)."""
    con = sqlite3.connect(DB)
    rows = con.execute("""SELECT name, domain, kind, description, source, rank, us_places FROM merchants WHERE split = ? AND kind IN ({})
                          ORDER BY CASE WHEN rank IS NOT NULL THEN rank WHEN us_places IS NOT NULL THEN 200000 - us_places * 10
                                        WHEN source = 'wikidata' THEN 300000 ELSE 900000 END""".format(",".join("?" * len(KIND_NAMES))),
                       (split, *KIND_NAMES)).fetchall()
    out, seen = [], set()
    for name, dom, kind, desc, src, rank, places in rows:
        key = re.sub(r"[^a-z0-9]", "", name.lower())
        if key and key not in seen and len(name) <= 60:
            seen.add(key); out.append(dict(name=name, domain=dom, kind=kind, description=desc))
        if len(out) >= n:
            break
    return out


def bank_string(m, rng):
    """A statement rendering of a merchant: its name, or (for an online one, half the time) its domain, through render_v2."""
    base = m["name"]
    if m["domain"] and rng.random() < 0.4:
        base = m["domain"].split(".")[0] if rng.random() < 0.5 else m["domain"]
    return render_v2(base, rng)


def row(m, rng, date):
    return f"{date.isoformat()} | {bank_string(m, rng).replace('|', '/')} | ${round(rng.lognormvariate(3.4, 0.9), 2):.2f} | {WD[date.weekday()]}"


if __name__ == "__main__":
    rng = random.Random(179)
    train = merchants("train", TOP)
    test_new = merchants("test", 2000)
    print(f"{len(train)} training merchants, {len(test_new)} held-out merchants", flush=True)
    names = list(KIND_NAMES.values())
    pool = [m for m in train for _ in range(EXPOSURES)]
    rng.shuffle(pool)
    out = PROCESSED / os.environ.get("OUT", "merchant_knowledge_v1_train.jsonl")
    k = 0
    with open(out, "w") as f:
        for s in range(0, len(pool) - ROWS, ROWS + 1):
            chunk = pool[s:s + ROWS + 1]
            opts = sorted(set(names), key=lambda _: rng.random())
            d0 = dt.date(rng.randint(2018, 2025), rng.randint(1, 12), 1)
            hist = "".join(f"Transaction: {row(m, rng, d0 + dt.timedelta(days=j))}\nCategory: {KIND_NAMES[m['kind']]}\n\n" for j, m in enumerate(chunk[:-1]))
            q = chunk[-1]
            ctx = "Categories: " + ", ".join(opts) + "\n\n" + hist + f"Transaction: {row(q, rng, d0 + dt.timedelta(days=ROWS))}"
            f.write(json.dumps(dict(context=ctx, options=opts, answer=opts.index(KIND_NAMES[q["kind"]]), spans=label_spans(ctx))) + "\n")
            k += 1
    print(f"{k} training episodes ({ROWS} merchants + 1 each) -> {out} ({out.stat().st_size / 1e6:.0f} MB)", flush=True)
    items = []
    for trained, ms in ((True, rng.sample(train, 2000)), (False, test_new)):
        for m in ms:
            opts = sorted(names, key=lambda _: rng.random())
            p = "Categories: " + ", ".join(opts) + f"\n\nTransaction: {row(m, rng, dt.date(2025, 6, 2))}\nCategory:"
            items.append(dict(id=f"MK:{'t' if trained else 'h'}:{len(items)}", options=[" " + o for o in opts], answer=opts.index(KIND_NAMES[m["kind"]]),
                              prompt=p, prompt_ctx=p, merchant=m["name"], kind=m["kind"], trained=trained, user=0))
    tp = PROCESSED / "merchant_knowledge_v1_test.json"
    tp.write_text(json.dumps(dict(name="merchant_knowledge_v1_test", version=1, n=len(items), items=items)))
    print(f"{len(items)} test items (2,000 trained merchants, {len(test_new)} held out) -> {tp}")
