"""A learned retriever for the REAL-6 records (PLAN step 37, REAL-7): from the card-statement string of a test transaction to the
merchant's fact-DB record, without the name lookup that sections 37 and 38 used as the retrieval oracle.

The index is the 240 records of `real6_v1` (one per merchant). The encoder is all-MiniLM-L6-v2 (`ai_experiments.retrieval.Retriever`),
zero-shot and tuned on pairs the DB alone can supply: templated statement renderings of each merchant's own name (`merchants.renderings`,
`bank_string`, the name, their normalised forms) -> the merchant's record. No user label and no test string is used; the test strings
are the 1,179 REAL-6 item strings, whose renderings draw fresh store numbers, dates and templates. Recall@1 / @5 overall, by known
chain / opaque merchant, by whether the full name survives in the string (against truncated or vowel-stripped), and on the DB-only
merchants; raw and normalised queries. The tuned retriever's top-5 per item goes to results/real6_retrieved.json, which
`exp_real6.py` reads for its `ret1` condition (CONDS=ret1 / ENC_CTX=ret): the record of the top-1 merchant, right or wrong.

A second index adds N_DECOYS opaque decoy merchants (fresh names from the same generator, records from the same pools, never
in any user's history) to the 240, so recall is also read against a DB of production size (5,240 records), where a truncated
KELVARR or vowel-stripped KLVRR has many near neighbours; the tuned encoder sees the decoys' renderings too (they are DB rows).

usage: uv run python scripts/exp_real6_retriever.py            EPOCHS=6 SEED=0 K_REND=8 N_DECOYS=5000
outputs: results/real6_retrieval.json, results/real6_retrieved.json, models/adapters/retriever_real6_minilm; tracker "real6_retrieval"
"""
import json
import os
import random
import time
from collections import defaultdict

import torch

from ai_experiments import merchants as M
from ai_experiments import real6 as R6
from ai_experiments import transactions as T
from ai_experiments.evals.tracker import Run
from ai_experiments.paths import ROOT
from ai_experiments.retrieval import Retriever

SEED = int(os.environ.get("SEED", "0"))
EPOCHS = int(os.environ.get("EPOCHS", "6"))
K_REND = int(os.environ.get("K_REND", "8"))
OUT = ROOT / "results" / "real6_retrieval.json"
OUT_ITEMS = ROOT / "results" / "real6_retrieved.json"

DOC = R6.load()
MS = T.load()["merchants"]
by_name = {m["name"]: m for m in MS}
docs = [DOC["fact_db"][m["name"]] for m in MS]
keys = [{m["name"]} for m in MS]
DB_ONLY = R6.db_only_merchants()
N_DECOYS = int(os.environ.get("N_DECOYS", "5000"))


def decoys(n, seed=77):
    """n opaque merchants with names the 240 do not use (the section 4 generator: prefix + suffix + tag), records from the pools."""
    rng = random.Random(seed)
    taken = set(by_name); out = []
    while len(out) < n:
        name = rng.choice(M._PREFIX) + rng.choice(M._SUFFIX) + rng.choice(M._TAG)
        if name in taken:
            continue
        taken.add(name)
        cat = rng.choice(M.CATEGORY_LIST); prods = rng.sample(M.CATEGORIES[cat], 3)
        out.append(dict(name=name, category=cat, products=prods, known=False, city=rng.choice(M._CITIES), n=rng.randint(1, 9999),
                        d=f"{rng.randint(1, 12):02d}/{rng.randint(1, 28):02d}", bank_tmpl=rng.randrange(len(M._BANK_TMPL)),
                        record=f"{name} is a store that sells {prods[0]}, {prods[1]} and {prods[2]}."))
    return out


def pairs(rng, ms):
    """(statement rendering -> record) from the DB's own names: templated renderings, the section 4 bank string, the bare name, and
    the normalised form of each; nothing from any user's history."""
    out = []
    for m in ms:
        rec = m["record"] if "record" in m else DOC["fact_db"][m["name"]]
        anchors = M.renderings(m, rng, K_REND) + [M.bank_string(m), m["name"], m["name"].upper()]
        anchors += [M.normalize(a) for a in anchors]
        out += [(a, rec) for a in dict.fromkeys(anchors)]
    return out


def full_name_in(text, m):
    return M._upper(m) in text.upper()


def recall_table(R, items, norm):
    q = [M.normalize(it["text"]) if norm else it["text"] for it in items]
    top = R.topk(q, 5)
    groups = defaultdict(list)
    for it, t in zip(items, top):
        m = by_name[it["merchant"]]
        hits = [it["merchant"] in R.keys[i] for i in t]
        g = ["all", "known" if m["known"] else "opaque", "full_name" if full_name_in(it["text"], m) else "truncated_or_abbr"]
        if it["merchant"] in DB_ONLY:
            g.append("db_only")
        for k in g:
            groups[k].append(hits)
    out = {}
    for g, hs in groups.items():
        out[f"{g}_recall@1"] = round(100 * sum(h[0] for h in hs) / len(hs), 1)
        out[f"{g}_recall@5"] = round(100 * sum(any(h) for h in hs) / len(hs), 1)
        out[f"{g}_n"] = len(hs)
    return out, top


items = DOC["items"]
cfg = dict(seed=SEED, epochs=EPOCHS, k_rend=K_REND, n_records=len(docs), n_items=len(items), real6_sha=DOC["sha256"], encoder=Retriever.__module__)
results = {}
with Run("real6_retrieval", model="sentence-transformers/all-MiniLM-L6-v2", config=cfg) as run:
    rng = random.Random(SEED)
    P = pairs(rng, MS)
    DEC = decoys(N_DECOYS)
    P_dec = P + pairs(rng, DEC)
    results["n_pairs"] = len(P); results["n_pairs_with_decoys"] = len(P_dec); results["n_decoys"] = len(DEC)
    for tag, train, extra in (("zero_shot", False, []), ("tuned", True, []), ("zero_shot_decoys", False, DEC), ("tuned_decoys", True, DEC)):
        t0 = time.time()
        R = Retriever(docs + [d["record"] for d in extra], keys + [{d["name"]} for d in extra], (P_dec if extra else P) if train else None,
                      epochs=EPOCHS if not extra else max(2, EPOCHS // 3), seed=SEED, log=f"real6 ({tag})")
        r = {}
        for norm in (False, True):
            rec, top = recall_table(R, items, norm)
            r.update({f"{'norm' if norm else 'raw'}_{k}": v for k, v in rec.items()})
            if tag == "tuned" and not norm:
                retrieved = {it["id"]: dict(top=[sorted(R.keys[i])[0] for i in t], hit1=it["merchant"] in R.keys[t[0]],
                                            rank_gold=next((k + 1 for k, i in enumerate(t) if it["merchant"] in R.keys[i]), None)) for it, t in zip(items, top)}
        r["minutes"] = round((time.time() - t0) / 60, 2)
        results[tag] = r; run.log(r, condition=tag)
        print(f"   {tag}: " + " ".join(f"{k}={v}" for k, v in r.items() if "recall" in k and ("all" in k or "db_only" in k or "truncated" in k)), flush=True)
        if train:
            R.save("real6_minilm" if not extra else "real6_minilm_decoys")
        del R; torch.cuda.empty_cache()
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(dict(config=cfg, **results), indent=2)); run.artifact(OUT)
    OUT_ITEMS.write_text(json.dumps(dict(retriever="retriever_real6_minilm", query="raw statement string", items=retrieved), indent=0)); run.artifact(OUT_ITEMS)
print(f"=== wrote {OUT.relative_to(ROOT)}, {OUT_ITEMS.relative_to(ROOT)}")
