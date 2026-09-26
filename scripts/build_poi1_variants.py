"""Variants of POI-1 (PLAN step 65) that change only the prompt, item for item, for scoring without retraining.

  kshots   kind-retrieved shots: up to 6 of the user's history places of the query's Overture basic category (what a places database
           tells a system about the query) replace frozen shots, the rest of the 24 frozen shots kept in order; when the history has none
           of that kind the prompt is unchanged. The history is the user's own, so this is retrieval over their past filings, not a label leak.
usage: uv run python scripts/build_poi1_variants.py kshots [--force]
"""
import json
import random
import sys

from ai_experiments import real6 as R6
from ai_experiments.paths import PROCESSED

SRC = PROCESSED / "poi1_v1.json"
K = 6


def kshots(doc):
    users = {u["user"]: u for u in doc["users"]}
    out = []
    for it in doc["items"]:
        u = users[it["user"]]; rng = random.Random(f"{it['id']}-kshots")
        by_text = {h["text"]: h for h in u["history"]}
        same = [h for h in u["history"] if h["basic"] == it["basic"]]
        rng.shuffle(same); same = same[:K]
        frozen = [by_text[t] for t in u["shots"] if by_text[t]["basic"] != it["basic"]]
        shots = same + frozen[:len(u["shots"]) - len(same)]
        rng.shuffle(shots)
        header = it["prompt"].split("\n\n", 1)[0] + "\n\n"
        demo = "".join(f"Transaction: {s['text']} | ${s['amount']:.2f} | {s['weekday']}\nCategory: {s['label']}\n\n" for s in shots)
        q = it["prompt"].rsplit("\n\n", 1)[1]
        assert q.startswith("Transaction: ") and q.endswith("Category:")
        record = f"Note: {it['record']}\n"
        out.append(dict(it, prompt=header + demo + q, prompt_ctx=header + demo + record + q, n_kind_shots=len(same)))
    return out


if __name__ == "__main__":
    which = sys.argv[1]
    dst = PROCESSED / f"poi1_v1_{which}.json"
    if dst.exists() and "--force" not in sys.argv:
        sys.exit(f"{dst} exists (frozen); pass --force to rebuild")
    doc = json.loads(SRC.read_text())
    items = {"kshots": kshots}[which](doc)
    # the users' frozen shots stay as they were: the scorer's flags read them, the prompts carry the retrieved ones
    new = dict(doc, version=f"v1_{which}", items=items, sha256=R6.sha256(items), variant_of=f"poi1_v1 ({doc['sha256'][:12]})")
    dst.write_text(json.dumps(new, indent=0, ensure_ascii=False))
    from collections import Counter
    print(len(items), "items; kind shots per item:", sorted(Counter(i["n_kind_shots"] for i in items).items()))
