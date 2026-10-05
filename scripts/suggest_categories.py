"""PLAN row 214 (exploratory, synthetic only): suggest a category scheme for a new household by clustering its spending.

Per held-out household (TEST_SEEDS, shared v4 test world), its categories hidden:
  points   the household's distinct payees from its outflows (hist_encoder.events: no inflows, transfers, splits), one per crowd key
           (build_crowd.crowd_key of the bank string; "payee:<clean payee>" when the key is empty), weighted by transaction count
           (spend reported). Display name: the payee's most common clean name. Gold: the payee's majority real category.
  features hist   the history encoder (HIST_ENC, hist_knn_v1) on the payee's clean name alone (not its transaction texts)
           crowd  the payee's crowd vector over row 189's k-means-64 category clusters (row 213's cache data/interim/r213: distinct other
                  households per cluster for the key, the household itself left out; under CROWD_K other households -> an "unknown" dim)
           amt    log median amount, share of the household's months present, log mean / std of days between, z-scored per household
           and concatenations (each block unit scale, then joined).
  methods  EVoC (evoc 0.3.1) with its defaults and with SMALL settings, each read at its selected layer (labels_), finest (L0),
           coarsest, and the layer whose cluster count is nearest the true count (oracle pick); k-means and agglomerative (average,
           cosine) with k = the true category count (oracle k).
  baselines B1 each payee's most common crowd cluster; B2 each payee's crowd-majority category name (uncovered payees are unclustered).
Metrics (mean over households): ARI / NMI against payee gold with payees repeated by count, unclustered points as singletons ("raw") or
attached to the nearest cluster centroid ("att"; baselines attach by the nearest covered payee in the hist space); clusters (non-noise)
vs true categories; edits = per true category (clusters its payees spread over - 1) + per cluster (true categories in it - 1), noise
singletons counting as clusters; unclustered share of transactions; purity = share of transactions whose payee's cluster has the
transaction's own category as its majority (noise = miss in raw).
Cluster names: the two category names other households most give to the cluster's payees (crowd, no LLM).
TREE_SEED: writes data/interim/r214/treemap_<seed>.json (EVoC tree coarse -> fine, names, sizes, top payees, real breakdown; the real
scheme groups -> categories) for TREE_FEATS / TREE_EVOC.
env: SHARED_WORLD=1 GROUPNAMES=1 REALSTYLE_V4=1 CROWD_KEY=v2; TEST_SEEDS (100000-100049), HIST_ENC (hist_knn_v1), DEVICE (cpu),
     CROWD_K (2), FEATS (hist,crowd,amt,hist+crowd,crowd+amt,hist+crowd+amt), SMALL (3,3,10: base_min_cluster_size,min_samples,n_neighbors),
     TREE_SEED (100003), TREE_FEATS (hist+crowd), TREE_EVOC (small).
usage: uv run --with evoc==0.3.1 --with matplotlib python scripts/suggest_categories.py
"""
import json
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from build_crowd import EMOJI, crowd_key  # noqa: E402
from build_crowd_clusters import seeds  # noqa: E402
from hist_encoder import events  # noqa: E402
from two_tower import households  # noqa: E402

from ai_experiments.paths import ROOT  # noqa: E402

TEST = seeds(os.environ.get("TEST_SEEDS", "100000-100049"))
HIST_ENC = os.environ.get("HIST_ENC", "hist_knn_v1")
DEVICE = os.environ.get("DEVICE", "cpu")
CROWD_K = int(os.environ.get("CROWD_K", "2"))
FEATS = os.environ.get("FEATS", "hist,crowd,amt,hist+crowd,crowd+amt,hist+crowd+amt").split(",")
SMALL = [int(x) for x in os.environ.get("SMALL", "3,3,10").split(",")]
TREE_SEED = int(os.environ.get("TREE_SEED", "100003"))
TREE_FEATS, TREE_EVOC = os.environ.get("TREE_FEATS", "hist+crowd"), os.environ.get("TREE_EVOC", "small")
R213 = ROOT / "data" / "interim" / "r213"
WORK = ROOT / "data" / "interim" / "r214"


def evoc_kw(name):
    if name == "small":
        b, m, n = SMALL
        return dict(base_min_cluster_size=b, min_samples=m, n_neighbors=n)
    return {}


# ---------------------------------------------------------------- crowd

class Crowd:
    """Row 213's filings (key, category text) of 800 test-world households: per key, distinct households per k-means-64 cluster and per
    category name; read for one household with that household left out."""

    def __init__(self):
        z = np.load(R213 / "cats_test_two_tower_v1_800.npz", allow_pickle=True)
        texts, owner, pairs = z["texts"], z["owner"], z["pairs"]
        self.keys = json.loads(str(z["keys"]))
        lab = np.load(R213 / "labels_test_800.npz")["kmeans64"]
        self.k = int(lab.max()) + 1
        names, disp, nid = [], {}, {}
        for t in texts:
            nm = t.split("; recently:")[0]
            nm = nm.split(": ", 1)[1] if ": " in nm else nm
            nm = EMOJI.sub("", nm).strip()
            key = " ".join(nm.lower().split())
            disp.setdefault(key, nm)
            names.append(nid.setdefault(key, len(nid)))
        self.name_disp = [disp[k] for k in sorted(nid, key=nid.get)]
        names = np.array(names, np.int32)
        k, t = pairs[:, 0], pairs[:, 1]
        self.t_cl = np.unique(np.stack([k, lab[t], owner[t]], 1), axis=0)  # (key, cluster, household)
        self.t_nm = np.unique(np.stack([k, names[t], owner[t]], 1), axis=0)  # (key, name, household)
        self.i_cl = np.searchsorted(self.t_cl[:, 0], np.arange(len(self.keys) + 1))
        self.i_nm = np.searchsorted(self.t_nm[:, 0], np.arange(len(self.keys) + 1))

    def vec(self, key, seed):
        kid = self.keys.get(key)
        v = np.zeros(self.k, np.float32)
        if kid is None:
            return v
        r = self.t_cl[self.i_cl[kid]:self.i_cl[kid + 1]]
        r = r[r[:, 2] != seed]
        np.add.at(v, r[:, 1], 1)
        return v if v.sum() >= CROWD_K else np.zeros_like(v)

    def names(self, key, seed):
        kid = self.keys.get(key)
        if kid is None:
            return Counter()
        r = self.t_nm[self.i_nm[kid]:self.i_nm[kid + 1]]
        r = r[r[:, 2] != seed]
        c = Counter(r[:, 1].tolist())
        return c if sum(c.values()) >= CROWD_K else Counter()


# ---------------------------------------------------------------- points

def points(b):
    """payee points of one budget: list of dicts (key, name, n, spend, cats Counter, gold, amts, days)."""
    import datetime as dt
    ev = events(b)
    by_id = {t["id"]: t for t in b["transactions"]}
    groups = {g["id"]: g["name"] for g in b["category_groups"]}
    cats = {c["id"]: c for c in b["categories"]}
    pts = {}
    for e in ev:
        t = by_id[e["id"]]
        key = crowd_key(t.get("import_payee_name_original") or "") or "payee:" + e["payee"].lower()
        p = pts.setdefault(key, dict(key=key, names=Counter(), cats=Counter(), amts=[], days=[]))
        p["names"][e["payee"]] += 1
        p["cats"][e["gold"]] += 1
        p["amts"].append(e["amt"])
        p["days"].append(dt.date.fromisoformat(t["date"]).toordinal())
    out = list(pts.values())
    for p in out:
        p["name"] = p["names"].most_common(1)[0][0]
        p["n"] = sum(p["cats"].values())
        p["spend"] = float(sum(p["amts"]))
        p["gold"] = p["cats"].most_common(1)[0][0]
    label = {cid: (groups.get(c["category_group_id"], ""), c["name"]) for cid, c in cats.items()}
    return out, label


def amt_feats(pts):
    d0 = min(min(p["days"]) for p in pts)
    d1 = max(max(p["days"]) for p in pts)
    months = max(1, (d1 - d0) / 30.44)
    rows = []
    for p in pts:
        d = np.sort(np.array(p["days"]))
        gaps = np.diff(d) if len(d) > 1 else np.array([d1 - d0 + 1.0])
        mon = len({(dd // 30.44) for dd in d})  # approx months present
        rows.append([np.log1p(np.median(p["amts"])), min(1.0, mon / months), np.log1p(gaps.mean()), np.log1p(gaps.std())])
    A = np.array(rows, np.float32)
    A = (A - A.mean(0)) / np.clip(A.std(0), 1e-6, None)
    return A / np.sqrt(A.shape[1])


def unit(X):
    return X / np.clip(np.linalg.norm(X, axis=1, keepdims=True), 1e-9, None)


# ---------------------------------------------------------------- metrics

def attach(lab, X):
    """noise points to the nearest cluster centroid (cosine)."""
    lab = lab.copy()
    m = lab < 0
    if not m.any() or m.all():
        if m.all():
            lab[:] = 0
        return lab
    Xn = unit(X)
    cl = np.unique(lab[~m])
    C = unit(np.stack([Xn[lab == c].mean(0) for c in cl]))
    lab[m] = cl[np.argmax(Xn[m] @ C.T, 1)]
    return lab


def score(lab, pts):
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
    lab = np.asarray(lab).copy()
    noise = lab < 0
    nxt = lab.max() + 1 if len(lab) else 0
    lab[noise] = np.arange(nxt, nxt + noise.sum())  # noise points as singletons
    w = np.array([p["n"] for p in pts])
    gold = [p["gold"] for p in pts]
    gi = {g: i for i, g in enumerate(sorted(set(gold)))}
    y = np.array([gi[g] for g in gold])
    yr, lr = np.repeat(y, w), np.repeat(lab, w)
    spread = defaultdict(set)
    mix = defaultdict(set)
    for a, g in zip(lab, y):
        spread[g].add(a)
        mix[a].add(g)
    edits = sum(len(s) - 1 for s in spread.values()) + sum(len(s) - 1 for s in mix.values())
    clus_cats = defaultdict(Counter)  # by transactions' own categories
    for p, a in zip(pts, lab):
        clus_cats[a].update(p["cats"])
    maj = {a: c.most_common(1)[0][0] for a, c in clus_cats.items()}
    good = sum(p["cats"][maj[a]] for p, a, nz in zip(pts, lab, noise) if not nz)
    return dict(ari=adjusted_rand_score(yr, lr), nmi=normalized_mutual_info_score(yr, lr), clusters=int(len(set(lab[~noise]))),
                true=len(gi), edits=edits, noise=float(w[noise].sum() / w.sum()), purity=good / w.sum())


def both(lab, X, pts):
    r = score(lab, pts)
    a = score(attach(np.asarray(lab), X), pts)
    r.update(ari_att=a["ari"], nmi_att=a["nmi"], edits_att=a["edits"], purity_att=a["purity"], clusters_att=a["clusters"])
    return r


# ---------------------------------------------------------------- clustering

def run_evoc(X, kw):
    import evoc
    c = evoc.EVoC(random_state=214, **kw)
    sel = np.asarray(c.fit_predict(X))
    return c, sel, [np.asarray(x) for x in c.cluster_layers_]


def name_of(idx, pts, pnames, crowd):
    tot = Counter()
    for i in idx:
        c = pnames[i]
        s = sum(c.values())
        if s:
            for k, v in c.items():
                tot[k] += pts[i]["n"] * v / s
    return [crowd.name_disp[k] for k, _ in tot.most_common(2)] or ["(no crowd names)"]


def treemap(seed, pts, label, c, layers, pnames, crowd, feats, kw):
    def node(idx, layer, cid):
        idx = sorted(idx, key=lambda i: -pts[i]["n"])
        real = Counter()
        for i in idx:
            for g, v in pts[i]["cats"].items():
                real[" : ".join(label[g])] += v
        return dict(layer=layer, cluster=cid, name=" / ".join(name_of(idx, pts, pnames, crowd)), payees=len(idx),
                    transactions=int(sum(pts[i]["n"] for i in idx)), spend=round(sum(pts[i]["spend"] for i in idx), 2),
                    top_payees=[[pts[i]["name"], pts[i]["n"]] for i in idx[:8]],
                    real_breakdown=[[k, v] for k, v in real.most_common()], children=[])

    tree = c.cluster_tree_
    nl = len(layers)

    def build(key):
        layer, cid = key
        n = node(np.where(layers[layer] == cid)[0], layer, int(cid))
        n["children"] = sorted((build(ch) for ch in tree.get(key, [])), key=lambda x: -x["transactions"])
        return n

    roots = sorted((build(ch) for ch in tree.get((nl, 0), [])), key=lambda x: -x["transactions"])
    covered = np.zeros(len(pts), bool)
    for L in layers:
        covered |= L >= 0
    un = node(np.where(~covered)[0], -1, -1)
    un["name"] = "(unclustered) " + un["name"]
    groups = defaultdict(lambda: defaultdict(lambda: dict(transactions=0, spend=0.0, payees=[])))
    for p in pts:
        for g, v in p["cats"].items():
            gr, nm = label[g]
            d = groups[gr][nm]
            d["transactions"] += v
            d["spend"] += p["spend"] * v / p["n"]
            d["payees"].append([p["name"], v])
    real = []
    for gr, cs in groups.items():
        kids = []
        for nm, d in cs.items():
            d["payees"] = sorted(d["payees"], key=lambda x: -x[1])[:8]
            d["spend"] = round(d["spend"], 2)
            kids.append(dict(name=nm, **d))
        kids.sort(key=lambda x: -x["transactions"])
        real.append(dict(name=gr, transactions=sum(k["transactions"] for k in kids), spend=round(sum(k["spend"] for k in kids), 2), categories=kids))
    real.sort(key=lambda x: -x["transactions"])
    out = dict(seed=seed, synthetic=True, features=feats, evoc_params=kw, n_payees=len(pts),
               transactions=int(sum(p["n"] for p in pts)), spend=round(sum(p["spend"] for p in pts), 2),
               layers=[int(len(set(L[L >= 0]))) for L in layers], evoc_tree=roots + ([un] if un["payees"] else []), real_scheme=real)
    WORK.mkdir(parents=True, exist_ok=True)
    f = WORK / f"treemap_{seed}.json"
    f.write_text(json.dumps(out, indent=1))
    return f, out


# ---------------------------------------------------------------- main

def main():
    import evoc
    from sklearn.cluster import AgglomerativeClustering, KMeans
    WORK.mkdir(parents=True, exist_ok=True)
    evoc.EVoC(random_state=0).fit_predict(np.random.default_rng(0).normal(size=(300, 20)).astype(np.float32))  # numba compile
    t0 = time.time()
    crowd = Crowd()
    print(f"crowd: {len(crowd.keys)} keys, {len(crowd.t_cl)} (key, cluster, household) rows, {len(crowd.name_disp)} names ({time.time() - t0:.0f}s)", flush=True)
    hh = list(zip(TEST, households("test", TEST)))
    per = {s: points(b) for s, b in hh}
    del hh
    f = WORK / f"hist_{HIST_ENC}.npz"
    names = sorted({p["name"] for pts, _ in per.values() for p in pts})
    if f.exists() and set(np.load(f, allow_pickle=True)["names"].tolist()) >= set(names):
        z = np.load(f, allow_pickle=True)
        E = dict(zip(z["names"].tolist(), z["E"]))
    else:
        from sentence_transformers import SentenceTransformer
        t0 = time.time()
        m = SentenceTransformer(str(ROOT / "models" / "encoders" / HIST_ENC), device=DEVICE)
        Em = m.encode(names, normalize_embeddings=True, batch_size=256, show_progress_bar=False).astype(np.float32)
        np.savez(f, names=np.array(names, object), E=Em)
        E = dict(zip(names, Em))
        print(f"embedded {len(names)} payee names on {DEVICE} in {time.time() - t0:.0f}s", flush=True)

    # payee-level ambiguity: the ceiling of any payee-level scheme
    amb = Counter()
    for pts, _ in per.values():
        for p in pts:
            amb["payees"] += 1; amb["tx"] += p["n"]
            amb["multi_payees"] += len(p["cats"]) > 1
            amb["multi_tx"] += p["n"] * (len(p["cats"]) > 1)
            amb["maj_tx"] += p["cats"][p["gold"]]
    print(f"\n{len(per)} households, {amb['payees'] / len(per):.0f} payee points and {amb['tx'] / len(per):.0f} outflows each on average; "
          f"{amb['multi_payees'] / amb['payees']:.1%} of payees ({amb['multi_tx'] / amb['tx']:.1%} of transactions) filed under 2+ categories; "
          f"ceiling of a payee-level scheme (transactions in their payee's majority category): {amb['maj_tx'] / amb['tx']:.1%}", flush=True)

    res = defaultdict(list)
    tree_done = None
    for n, (seed, (pts, label)) in enumerate(per.items()):
        t0 = time.time()
        H = np.stack([E[p["name"]] for p in pts])
        Cv = np.stack([crowd.vec(p["key"], seed) for p in pts])
        cov = Cv.sum(1) > 0
        Cf = unit(np.concatenate([Cv, (~cov)[:, None].astype(np.float32)], 1))
        A = amt_feats(pts)
        blocks = dict(hist=H, crowd=Cf, amt=A)
        pnames = [crowd.names(p["key"], seed) for p in pts]
        k_true = len({p["gold"] for p in pts})
        res["_cov"].append(dict(cov=float(sum(p["n"] for p, c in zip(pts, cov) if c) / sum(p["n"] for p in pts))))
        # baselines (uncovered payees unclustered; attached by nearest covered payee in hist space)
        b1 = np.where(cov, Cv.argmax(1), -1)
        nm_top = [c.most_common(1)[0][0] if c else -1 for c in pnames]
        ids = {v: i for i, v in enumerate(sorted(set(nm_top) - {-1}))}
        b2 = np.array([ids.get(v, -1) for v in nm_top])
        for bname, lab in (("B1 crowd cluster", b1), ("B2 crowd name", b2)):
            r = score(lab, pts)
            la = lab.copy()
            if (la >= 0).any() and (la < 0).any():
                la[la < 0] = la[la >= 0][np.argmax(H[la < 0] @ H[la >= 0].T, 1)]
            a = score(la, pts)
            r.update(ari_att=a["ari"], nmi_att=a["nmi"], edits_att=a["edits"], purity_att=a["purity"], clusters_att=a["clusters"])
            res[("baseline", bname)].append(r)
        for fs in FEATS:
            X = np.concatenate([blocks[b] for b in fs.split("+")], 1).astype(np.float32)
            Xn = unit(X)
            res[(fs, "k-means k=true")].append(both(KMeans(n_clusters=k_true, n_init=4, random_state=214).fit_predict(Xn), X, pts))
            res[(fs, "agglo avg cos k=true")].append(both(AgglomerativeClustering(n_clusters=k_true, metric="cosine", linkage="average").fit_predict(X), X, pts))
            for ev in ("default", "small"):
                kw = evoc_kw(ev)
                c, sel, layers = run_evoc(X, kw)
                cnt = [len(set(L[L >= 0])) for L in layers]
                kc = int(np.argmin([abs(x - k_true) for x in cnt]))
                for tag, L in (("selected", sel), ("L0 finest", layers[0]), ("coarsest", layers[-1]), ("layer nearest k", layers[kc])):
                    res[(fs, f"EVoC {ev} {tag}")].append(both(L, X, pts))
                res[(fs, f"EVoC {ev} layers")].append(dict(layers=len(layers), counts=cnt))
                if seed == TREE_SEED and fs == TREE_FEATS and ev == TREE_EVOC:
                    tree_done = treemap(seed, pts, label, c, layers, pnames, crowd, fs, kw)
        print(f"  {seed}: {len(pts)} payees, {k_true} categories, {time.time() - t0:.0f}s", flush=True)

    cov = np.mean([r["cov"] for r in res.pop("_cov")])
    print(f"\ncrowd vector covers {cov:.1%} of transactions (payee keys with >= {CROWD_K} other households)")
    print(f"EVoC small = base_min_cluster_size {SMALL[0]}, min_samples {SMALL[1]}, n_neighbors {SMALL[2]}; default = 5, 5, 15")
    for (fs, meth), rs in res.items():
        if meth.endswith("layers"):
            print(f"  {fs} {meth}: mean {np.mean([r['layers'] for r in rs]):.1f} layers; example counts {rs[0]['counts']}")
    print("\n| features | method | ARI | NMI | clusters / true | edits | unclustered tx | purity | ARI att | NMI att | edits att | purity att |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for (fs, meth), rs in res.items():
        if meth.endswith("layers"):
            continue
        m = {k: np.mean([r[k] for r in rs]) for k in rs[0]}
        print(f"| {fs} | {meth} | {m['ari']:.3f} | {m['nmi']:.3f} | {m['clusters']:.0f} / {m['true']:.0f} | {m['edits']:.0f} | {m['noise']:.1%} | "
              f"{m['purity']:.1%} | {m['ari_att']:.3f} | {m['nmi_att']:.3f} | {m['edits_att']:.0f} | {m['purity_att']:.1%} |", flush=True)
    if tree_done:
        f, out = tree_done
        print(f"\ntreemap -> {f.relative_to(ROOT)}: {out['n_payees']} payees, layers {out['layers']}; top nodes:")
        for nd in out["evoc_tree"][:12]:
            print(f"  [{nd['transactions']} tx, {nd['payees']} payees] {nd['name']}: {[p for p, _ in nd['top_payees'][:4]]}; real {nd['real_breakdown'][:3]}")


if __name__ == "__main__":
    main()
