"""The user's own filing against the database and the name (PLAN step 85, REAL-15): override_v1. For fold 0's held-out users, fact-DB
merchants placed in two of the user's 24 shots and asked about again:

  agree     the two shots file the merchant under the user's category for its DB category (the lookup and the DB agree); gold that
  override  the two shots file it under another of the user's categories (drawn per user and merchant); gold that category: the
            user's own filing must beat the database and, for misleading names, the name

Merchants: REAL-6's DB-only merchants (in no training user's history; real chains and opaque names) and mislead_v1's in-DB misleading
names and neutral twins (in the DB when a model is trained with MISLEAD=mislead_v1). `db_idx` is the option of the DB category's user
category and `decoy_idx` that of the category a misleading name suggests (-1 when not a separate option), for the share pulled to each.
usage: uv run python scripts/build_override.py [--force]
"""
import json
import math
import random
import sys

from ai_experiments import real6 as R6
from ai_experiments import transactions as T
from ai_experiments.paths import PROCESSED

DST = PROCESSED / "override_v1.json"


def row(m, rng):
    mu, sig = T.AMOUNT[m["category"]]
    return dict(text=T.render(m, rng), amount=round(math.exp(rng.gauss(mu, sig)), 2), weekday=rng.choice(T.WEEKDAYS))


def line(h):
    return f"Transaction: {h['text']} | ${h['amount']:.2f} | {h['weekday']}"


if __name__ == "__main__":
    if DST.exists() and "--force" not in sys.argv:
        sys.exit(f"{DST} exists (frozen); pass --force to rebuild")
    doc = R6.load()
    base = {m["name"]: dict(m, kind="real6_known" if m["known"] else "real6_opaque", decoy=None) for m in T.load()["merchants"]}
    ms = [base[n] for n in sorted(R6.db_only_merchants())]
    ml = json.loads((PROCESSED / "mislead_v1.json").read_text())
    ms += [dict(m, kind=m["kind"]) for m in ml["merchants"] if m["in_db"]]
    items = []
    for u in doc["users"]:
        if u["user"] % 4:
            continue
        cats = u["categories"]; names = [c["name"] for c in cats]
        one = {}
        for c in cats:
            for s in c["standard"]:
                one.setdefault(s, []).append(c)
        prompt0 = next(it for it in doc["items"] if it["user"] == u["user"])["prompt"]
        header, rest = prompt0.split("\n\n", 1)
        shots = rest.split("\n\n")[:-1]
        for m in ms:
            owners = one.get(m["category"], [])
            if len(owners) != 1 or "split" in owners[0]:
                continue
            db_idx = names.index(owners[0]["name"])
            dec = one.get(m["decoy"], []) if m.get("decoy") else []
            decoy_idx = names.index(dec[0]["name"]) if len(dec) == 1 and "split" not in dec[0] and dec[0]["name"] != names[db_idx] else -1
            r = random.Random(f"override-{u['user']}-{m['name']}")
            alt = r.choice([k for k, c in enumerate(cats) if k != db_idx and "split" not in c])
            for cond, gold in (("agree", db_idx), ("override", alt)):
                rr = random.Random(f"{u['user']}-{m['name']}-{cond}")
                demo = list(shots)
                for j in rr.sample(range(len(demo)), 2):
                    demo[j] = line(row(m, rr)) + f"\nCategory: {names[gold]}"
                q = row(m, rr)
                prompt = header + "\n\n" + "\n\n".join(demo) + "\n\n" + line(q) + "\nCategory:"
                items.append(dict(id=f"OV_{m['kind']}_{cond}:{u['user']}:{m['name']}", level=f"OV_{m['kind']}_{cond}", user=u["user"], merchant=m["name"],
                                  known=m.get("known", False), text=q["text"], amount=q["amount"], weekday=q["weekday"], prompt=prompt, prompt_ctx=prompt,
                                  options=[" " + n for n in names], answer=gold, name_type=cats[gold]["name_type"], seen=True, mkind=m["kind"], cond=cond,
                                  db_idx=db_idx, decoy_idx=decoy_idx))
    out = dict(version="override_v1", variant_of=f"real6 v1 ({doc['sha256'][:12]}) + mislead_v1", items=items, sha256=R6.sha256(items))
    DST.write_text(json.dumps(out, indent=0, ensure_ascii=False))
    from collections import Counter
    print(len(items), "items:", dict(Counter(i["level"] for i in items)))
