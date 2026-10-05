"""PLAN row 213 (a)-(b): EVoC (Tutte Institute, evoc 0.3.x, BSD-2-Clause) against row 189's k-means 64 for the crowd's category clusters.

The texts are build_crowd_clusters.py's: one per (household, category) with filings, "<Group>: <Name>; recently: <3 payees>", embedded with
ENC (two_tower_v1) and clustered. (a) Intrinsic: every clustering against the generator's own purpose for each category (canon(): read off
realstyle.Household's construction, which records each category's role: an EVERYDAY key by its kinds list, a person's fun / clothes / groom /
treat / work, a kid phase category, pet, trip, holiday, property bill, named service, wish, savings, catch-all, store-named): ARI, NMI,
purity on the clustered points, the noise share, the cluster count, wall time. (b) Extrinsic proxy: on held-out households (TEST_SEEDS),
each payee's crowd vector over clusters as real_budget_eval.py SIM_SRC=behav builds it (the payee name's crowd keys, household counts per
cluster with the household itself left out, no vector under CROWD_K=2 other households); for each first-time-payee transaction the
household's earlier payees with a vector ranked by cosine; hit@1 / hit@3 = the nearest one / three payees' latest category is the gold one.
References: the household's most-used category so far, and the nearest earlier payee by character n-grams (real_budget_eval's SIM2 clean).

env: SHARED_WORLD=1 GROUPNAMES=1 REALSTYLE_V4=1 CROWD_KEY=v2; SPLIT (test), SEEDS (200000-200699,100000-100099), TEST_SEEDS (100000-100049),
ENC (two_tower_v1), K (64), DEVICE (cpu), EVOC (default,b64: EVoC runs; "b64" = base_n_clusters=64).
usage: uv run --with evoc --with matplotlib python scripts/evoc_clusters.py   (evoc imports matplotlib without declaring it)
Intermediates (texts, purposes, embeddings, filings, labels) go to data/interim/r213/ (gitignored) and are reused when present.
"""
import json
import os
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from build_crowd import crowd_key  # noqa: E402
from build_crowd_clusters import seeds  # noqa: E402
from two_tower import _clean, households  # noqa: E402

from ai_experiments import realstyle as R  # noqa: E402
from ai_experiments.paths import ROOT  # noqa: E402

SPLIT = os.environ.get("SPLIT", "test")
ENC, K = os.environ.get("ENC", "two_tower_v1"), int(os.environ.get("K", "64"))
DEVICE = os.environ.get("DEVICE", "cpu")
CROWD_K = int(os.environ.get("CROWD_K", "2"))
EVOC_RUNS = os.environ.get("EVOC", "default,b64").split(",")
WORK = ROOT / "data" / "interim" / "r213"


def canon(seed, split):
    """{category id: the generator's purpose key}, from the household's construction (no simulation: the ids match its budget's)."""
    h = R.Household(seed, split)
    out = {h.rta: "rta"}
    by_kinds = {tuple(k): key for key, _, k, _, _ in R.EVERYDAY}
    kid = {("big_box", "pharmacy", "grocery"): "kid:diapers", ("clothing", "secondhand"): "kid:baby_clothes",
           ("kids", "education", "fitness"): "kid:school", ("clothing",): "kid:clothes", ("hobby", "big_box"): "kid:other"}
    for c in h.cats.values():  # everyday purposes carry their EVERYDAY kinds list
        if c.get("kinds") and tuple(c["kinds"]) in by_kinds and c["category_group_id"] in {h.groups.get(g) for g in
                                                                                           ("Everyday Expenses", "Monthly Bills", "House expenses", "Planning ahead", "Giving")}:
            out[c["id"]] = "everyday:" + by_kinds[tuple(c["kinds"])]
    for cid in h.catchall:
        out[cid] = "catchall"
    for cid in h.store_named.values():
        out[cid] = "store_named"
    for p in h.person.values():
        for role in ("fun", "clothes", "groom", "treat", "work"):
            if p[role]:
                out[p[role]] = "person:" + role
    for k in h.kid_cats:
        out[k["cid"]] = kid[tuple(k["kinds"])]
    for cid in h.pet_cats:
        out[cid] = "pet"
    for t in h.trips:
        out[t["cid"]] = "trip"
    out[h.travel_other] = "travel_other"
    hol = {(11, 15, 12, 31): "christmas", (10, 1, 10, 31): "halloween", (11, 18, 11, 28): "thanksgiving", (3, 25, 4, 15): "easter"}
    for x in h.holidays:
        out[x["cid"]] = "holiday:" + hol[x["window"]]
    for p in h.properties:
        for k, cid in p["cats"].items():
            out[cid] = "property:" + k
    for s in h.services:
        out[s["cid"]] = "service:" + s["kind"] + ":" + re.sub(r"\W+", "_", h.cats[s["cid"]]["name"].split(" ")[0].lower())
    for cid, _ in h.wish:
        out[cid] = "wish"
    for c in h.cats.values():
        if c["id"] not in out and c["name"] in ("Long Term Investments/Savings", "Next months money", "Emergency fund", "New Car", "Buying a house"):
            out[c["id"]] = "saving:" + c["name"]
    return out


def build(hh):
    """texts, purposes, owners per (household, category with filings); crowd filings as distinct (key, text index) pairs."""
    texts, purpose, owner, pairs, keys = [], [], [], set(), {}
    unmapped = Counter()
    for n, (seed, b) in enumerate(zip(hh, households(SPLIT, hh))):
        can = canon(seed, SPLIT)
        groups = {g["id"]: g["name"] for g in b["category_groups"]}
        cats = {c["id"]: c for c in b["categories"]}
        pname = {p["id"]: p["name"] for p in b["payees"]}
        rta = next(c["id"] for c in b["categories"] if c["name"] == "Inflow: Ready to Assign")
        recent, idx = defaultdict(list), {}
        for t in sorted(b["transactions"], key=lambda t: t["date"]):
            if t["category_id"] not in cats or t["category_id"] == rta or t.get("kind") == "p2p":
                continue
            p = _clean(pname.get(t["payee_id"], ""))
            recent[t["category_id"]] = [p] + [x for x in recent[t["category_id"]] if x != p][:4]
            if t["category_id"] not in idx:
                idx[t["category_id"]] = len(texts)
                texts.append(None); owner.append(seed)
                pu = can.get(t["category_id"])
                if pu is None:
                    unmapped[cats[t["category_id"]]["name"]] += 1
                purpose.append(pu or "unmapped")
            key = crowd_key(t["import_payee_name_original"])
            if key:
                pairs.add((keys.setdefault(key, len(keys)), idx[t["category_id"]]))
        for cid, k in idx.items():
            c = cats[cid]
            texts[k] = f'{_clean(groups.get(c["category_group_id"], ""))}: {_clean(c["name"])}' + ("; recently: " + ", ".join(recent[cid][:3]) if recent[cid] else "")
        if n % 100 == 0:
            print(f"  {n} households, {len(texts)} categories, {len(pairs)} filings", flush=True)
    if unmapped:
        print("unmapped categories:", unmapped.most_common(10))
    pairs = np.array(sorted(pairs), dtype=np.int32)
    return texts, purpose, np.array(owner, np.int32), pairs, keys


def purity(y, lab):
    m = lab >= 0
    tab = defaultdict(Counter)
    for a, b in zip(lab[m], y[m]):
        tab[a][b] += 1
    return sum(c.most_common(1)[0][1] for c in tab.values()) / max(1, m.sum())


def intrinsic(name, lab, y, secs):
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
    m = lab >= 0
    return dict(clustering=name, clusters=int(len(set(lab[m]))), noise=float(1 - m.mean()), ari=adjusted_rand_score(y[m], lab[m]),
                nmi=normalized_mutual_info_score(y[m], lab[m]), purity=purity(y, lab), secs=secs)


def first_time_items(b):
    """[(transaction index, payee name, gold category, date)] in real_budget_eval's order, and the payee -> crowd keys map."""
    cats = {c["id"] for c in b["categories"]}
    rta = next(c["id"] for c in b["categories"] if c["name"] == "Inflow: Ready to Assign")
    pname = {p["id"]: p["name"] for p in b["payees"]}
    rows = [t for t in b["transactions"] if not t.get("deleted") and t.get("approved") and t["category_id"] in cats and t["category_id"] != rta]
    rows.sort(key=lambda t: (t["date"], t["id"]))
    pkeys = defaultdict(set)
    for t in rows:
        p = (pname.get(t["payee_id"], "") or "").replace("|", "/").replace("\n", " ").strip()
        t["_p"] = p
        k = crowd_key(t["import_payee_name_original"] or p)
        if k:
            pkeys[p].add(k)
    return rows, pkeys


def ngram_clean(n):  # real_budget_eval.py SIM2's clean
    n = re.sub(r"^(sale|return|sq|tst|pos|debit|purchase|paypal)\b\W*", "", n.lower())
    return re.sub(r"\s+", " ", re.sub(r"[^a-z ]", " ", n)).strip()


def extrinsic(clusterings, keys, owner, pairs, test_seeds):
    """hit@1 / hit@3 of the nearest earlier payee by crowd vector (one clustering each) on first-time-payee transactions."""
    import scipy.sparse as sp
    from sklearn.feature_extraction.text import TfidfVectorizer
    n_keys = len(keys)
    # per clustering: (key, cluster) -> number of distinct households, as a sparse matrix
    counts = {}
    for name, lab in clusterings.items():
        cl = lab[pairs[:, 1]]
        m = cl >= 0
        trip = np.unique(np.stack([pairs[m, 0], cl[m], owner[pairs[m, 1]]], 1), axis=0)
        counts[name] = sp.csr_matrix((np.ones(len(trip), np.float32), (trip[:, 0], trip[:, 1])), shape=(n_keys, int(lab.max()) + 1))
    own_pairs = defaultdict(list)  # household -> its (key, text) pairs
    tset = set(test_seeds)
    for k, t in pairs:
        if owner[t] in tset:
            own_pairs[int(owner[t])].append((int(k), int(t)))
    stats = {name: Counter() for name in list(clusterings) + ["ngram", "prior"]}
    for seed, b in zip(test_seeds, households(SPLIT, test_seeds)):
        rows, pkeys = first_time_items(b)
        names = sorted(pkeys.keys() | {t["_p"] for t in rows})
        pos = {p: i for i, p in enumerate(names)}
        vecs = {}
        for name, lab in clusterings.items():
            C = counts[name]
            mine = defaultdict(set)
            for k, t in own_pairs[seed]:
                if lab[t] >= 0:
                    mine[k].add(int(lab[t]))
            X = np.zeros((len(names), C.shape[1]), np.float32)
            for p, ks in pkeys.items():
                for k in ks:
                    kid = keys.get(k)
                    if kid is None:
                        continue
                    row = C.getrow(kid)
                    X[pos[p], row.indices] += row.data
                    for c in mine.get(kid, ()):
                        X[pos[p], c] -= 1  # the household itself left out
            X[X.sum(1) < CROWD_K] = 0
            X /= np.clip(np.linalg.norm(X, axis=1, keepdims=True), 1e-9, None)
            vecs[name] = X
        G = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5)).fit_transform([ngram_clean(n) for n in names])
        vecs["ngram"] = G
        last, seen, used = {}, [], Counter()
        for t in rows:
            p, gold = t["_p"], t["category_id"]
            if p not in last and seen:
                q = pos[p]
                cand = np.array(seen)
                for name, X in vecs.items():
                    s = stats[name]
                    s["n"] += 1
                    if name == "ngram":
                        sims = (X[cand] @ X[q].T).toarray().ravel()
                        ok = np.ones(len(cand), bool)
                    else:
                        if not X[q].any():
                            continue
                        ok = X[cand].any(1)
                        if not ok.any():
                            continue
                        sims = X[cand] @ X[q]
                    s["covered"] += 1
                    order = [j for j in np.argsort(-sims) if ok[j]][:3]
                    got = [last[names[cand[j]]] for j in order]
                    s["hit1"] += got[0] == gold
                    s["hit3"] += gold in got
                stats["prior"]["n"] += 1; stats["prior"]["covered"] += 1
                stats["prior"]["hit1"] += used.most_common(1)[0][0] == gold
                stats["prior"]["hit3"] += gold in [c for c, _ in used.most_common(3)]
            if p not in last:
                seen.append(pos[p])
            last[p] = gold
            used[gold] += 1
        del vecs
    return stats


def main():
    assert R.SHARED and R.V4, "the shared v4 world: SHARED_WORLD=1 REALSTYLE_V4=1"
    WORK.mkdir(parents=True, exist_ok=True)
    hh = seeds(os.environ.get("SEEDS", "200000-200699,100000-100099"))
    test_seeds = seeds(os.environ.get("TEST_SEEDS", "100000-100049"))
    f = WORK / f"cats_{SPLIT}_{ENC}_{len(hh)}.npz"
    if f.exists():
        z = np.load(f, allow_pickle=True)
        texts, purpose, owner, pairs, X = list(z["texts"]), list(z["purpose"]), z["owner"], z["pairs"], z["X"]
        keys = json.loads(str(z["keys"]))
        print(f"loaded {f.name}: {len(texts)} categories")
    else:
        t0 = time.time()
        texts, purpose, owner, pairs, keys = build(hh)
        print(f"built {len(texts)} category texts from {len(hh)} households in {time.time() - t0:.0f}s; {len(keys)} crowd keys, {len(pairs)} filings")
        from sentence_transformers import SentenceTransformer
        t0 = time.time()
        model = SentenceTransformer(str(ROOT / "models" / "encoders" / ENC), device=DEVICE)
        X = model.encode(texts, normalize_embeddings=True, batch_size=256, show_progress_bar=False).astype(np.float32)
        print(f"embedded on {DEVICE} in {time.time() - t0:.0f}s")
        np.savez(f, texts=np.array(texts, object), purpose=np.array(purpose, object), owner=owner, pairs=pairs, X=X, keys=json.dumps(keys))
    pv = Counter(purpose)
    print(f"{len(pv)} purposes; largest: {pv.most_common(6)}; unmapped {pv.get('unmapped', 0)}")
    y = np.array([hash(p) for p in purpose])

    clusterings, rows = {}, []
    from sklearn.cluster import KMeans
    t0 = time.time()
    lab = KMeans(n_clusters=K, n_init=4, random_state=189).fit_predict(X)
    secs = time.time() - t0
    clusterings[f"kmeans{K}"] = lab
    rows.append(intrinsic(f"k-means {K}", lab, y, secs))
    import evoc
    evoc.EVoC(random_state=213).fit_predict(X[:2000])  # numba compiles once; timed runs below exclude it
    for run in EVOC_RUNS:
        kw = dict(base_n_clusters=K) if run == "b64" else {}
        t0 = time.time()
        c = evoc.EVoC(random_state=213, **kw)
        c.fit_predict(X)
        secs = time.time() - t0
        for i, layer in enumerate(c.cluster_layers_):
            layer = np.asarray(layer)
            name = f"evoc_{run}_L{i}"
            clusterings[name] = layer
            rows.append(intrinsic(f"EVoC {run} layer {i}", layer, y, secs))
        print(f"EVoC {run}: {len(c.cluster_layers_)} layers in {secs:.0f}s; {len(c.duplicates_)} duplicate pairs; tree nodes {len(c.cluster_tree_)}")
    np.savez(WORK / f"labels_{SPLIT}_{len(hh)}.npz", **clusterings)

    print("\n**(a) intrinsic: clusterings of", len(texts), "category texts against", len(pv), "generator purposes (on clustered points)**\n")
    print("| clustering | clusters | noise | ARI | NMI | purity | secs |\n|---|---|---|---|---|---|---|")
    for r in rows:
        print(f"| {r['clustering']} | {r['clusters']} | {r['noise']:.1%} | {r['ari']:.3f} | {r['nmi']:.3f} | {r['purity']:.3f} | {r['secs']:.0f} |", flush=True)

    t0 = time.time()
    stats = extrinsic(clusterings, keys, owner, pairs, test_seeds)
    print(f"\n**(b) first-time payees in {len(test_seeds)} held-out households: nearest earlier payee's category ({time.time() - t0:.0f}s)**\n")
    print("| neighbours by | n | covered | hit@1 all | hit@3 all | hit@1 covered | hit@3 covered |\n|---|---|---|---|---|---|---|")
    for name, s in stats.items():
        n, cv = max(1, s["n"]), max(1, s["covered"])
        print(f"| {name} | {s['n']} | {s['covered'] / n:.1%} | {s['hit1'] / n:.1%} | {s['hit3'] / n:.1%} | {s['hit1'] / cv:.1%} | {s['hit3'] / cv:.1%} |")


if __name__ == "__main__":
    main()
