"""PLAN step 189 (owner, 2026-10-04: "payee X is often categorized as category clusters x, y, and z, and clusters x, y, and z are the way
payees a, b, and c are often categorized"): a world's categories clustered across households, and each bank string's filings counted
per cluster. Every category of every household in the world is written as row 187's category text ("<Group>: <Name>; recently: <the three
payees last filed under it>", as of the household's last day), embedded with the two-tower encoder (ENC) and clustered by k-means (K);
then each bank string (build_crowd.crowd_key) gets, per cluster, the households that filed it there. real_budget_eval.py SIM_SRC=behav
turns that into a behaviour vector per payee (the household being read left out) and picks similar payees by cosine between vectors.
Writes data/processed/<OUT>: {"k": K, "households": [...], "keys": {key: {cluster: [household seeds]}}, "examples": {cluster: [3 names]}}.
env: SPLIT (test), SEEDS ("200000-200699,100000-100099" style ranges or a list), ENC (two_tower_v1), K (64), OUT (realstyle_crowdclus_test.json).
usage: SHARED_WORLD=1 GROUPNAMES=1 uv run python scripts/build_crowd_clusters.py
"""
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from build_crowd import crowd_key  # noqa: E402
from two_tower import _clean  # noqa: E402

from ai_experiments import realstyle as R  # noqa: E402
from ai_experiments.paths import PROCESSED, ROOT  # noqa: E402

SPLIT = os.environ.get("SPLIT", "test")
ENC, K = os.environ.get("ENC", "two_tower_v1"), int(os.environ.get("K", "64"))
OUT = PROCESSED / os.environ.get("OUT", "realstyle_crowdclus_test.json")


def seeds(spec):
    out = []
    for part in spec.split(","):
        a, _, b = part.partition("-")
        out += list(range(int(a), int(b) + 1)) if b else [int(a)]
    return out


if __name__ == "__main__":
    assert R.SHARED, "a shared world: SHARED_WORLD=1"
    import torch
    from sentence_transformers import SentenceTransformer
    from sklearn.cluster import KMeans
    hh = seeds(os.environ.get("SEEDS", "200000-200699,100000-100099"))
    texts, owner, filings = [], [], []  # one text per (household, category); filings: (household, key, category index)
    for n, seed in enumerate(hh):
        h = R.household(seed, SPLIT)
        groups = {i: g for g, i in h.groups.items()}
        recent = defaultdict(list)
        idx = {}
        for t in sorted(h.txs, key=lambda t: t["date"]):
            c = h.cats.get(t["category_id"])
            if not c or t["category_id"] == h.rta or t.get("kind") == "p2p":
                continue
            p = _clean(next((nm for nm, i in h.payees.items() if i == t["payee_id"]), ""))
            recent[t["category_id"]] = [p] + [x for x in recent[t["category_id"]] if x != p][:4]
            if t["category_id"] not in idx:
                idx[t["category_id"]] = len(texts)
                texts.append(None); owner.append(seed)
            filings.append((seed, crowd_key(t["import_payee_name_original"]), idx[t["category_id"]]))
        for cid, k in idx.items():
            c = h.cats[cid]
            texts[k] = f'{_clean(groups.get(c["category_group_id"], ""))}: {_clean(c["name"])}' + ("; recently: " + ", ".join(recent[cid][:3]) if recent[cid] else "")
        if n % 100 == 0:
            print(f"  {n} households, {len(texts)} categories", flush=True)
    model = SentenceTransformer(str(ROOT / "models" / "encoders" / ENC), device="cuda" if torch.cuda.is_available() else "cpu")
    X = model.encode(texts, normalize_embeddings=True, batch_size=512, show_progress_bar=False)
    lab = KMeans(n_clusters=K, n_init=4, random_state=189).fit_predict(X)
    keys = defaultdict(lambda: defaultdict(set))
    for seed, key, k in filings:
        if key:
            keys[key][int(lab[k])].add(seed)
    ex = defaultdict(list)
    for t, l in zip(texts, lab):
        if len(ex[int(l)]) < 3:
            ex[int(l)].append(t.split(";")[0])
    OUT.write_text(json.dumps(dict(k=K, households=hh, keys={k: {str(c): sorted(s) for c, s in v.items()} for k, v in keys.items()},
                                   examples={str(c): v for c, v in sorted(ex.items())})))
    print(f"{len(texts)} categories from {len(hh)} households in {K} clusters; {len(keys)} keys -> {OUT} ({OUT.stat().st_size / 1e6:.0f} MB)")
    for c in range(min(K, 12)):
        print(f"  cluster {c}: {ex[c]}")
