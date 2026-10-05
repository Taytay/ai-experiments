"""PLAN step 204 (owner, 2026-10-05: "apply all of their learnings to our approach"; references/papers INDEX thread 7): several prototypes
per category, training-free, over a history encoder's vectors, against the MaxSim reader (hist_fast: mean of the 3 best cosines among the
category's earlier filings). Each event scores the household's visible categories from filings on earlier days only.
Readers (score = max over a category's prototypes of cos / TAU; a category with no earlier filing scores -1 / TAU, as MaxSim):
  medoid       PinnerSage (2007.03634): Ward clustering of the category's earlier filings (scipy, Euclidean on unit vectors), cut at
               merge height ALPHA, at most MAXC clusters; each cluster's medoid (the member with the largest summed cosine to the rest)
  centroid     the same clusters, each as its normalised mean
  medoid+imp   medoid score + IMP_W * log importance, importance = sum over the cluster's members of exp(-lambda * age in calendar days),
               one arm per lambda in LAMBDAS. The current day's exp(-lambda * T_now) is common to every category, so it cancels in the
               ranking: log importance is stored as logsumexp(lambda * T_i) when the category is clustered, valid on every later day.
  fp K         Multi-Embedding Retrieval (2506.23060): validity-aware farthest-point initialisation of up to K prototypes (first: the
               filing nearest the category mean; next: the valid filing farthest, in max cosine, from those chosen), then FP_ITERS rounds
               of single-assignment refinement (hard assignment, normalised mean: DCM's routing without the learned MLP). Invalid rows
               cannot be chosen as a start: hist_encoder.events already drops transfers, split parents and inflows (refunds), so the
               only rows masked here are those with an empty payee.
A category's prototypes are recomputed only on a day when it has gained filings since it was last clustered (cached otherwise); each
clustering uses at most HIST_CAP of its most recent filings (PinnerSage's 90-day window), so importance at lambda 0 counts that window.
Output: one markdown table, % top-1 / top-3 / top-10, all and first-time payee, n (ranks as hist_train2.read: ties count against the gold).
env: ENC (hist_knn_v1, under models/encoders), TEST_SEEDS (100000-100049), LAMBDAS (0,0.01,0.05), K (3,5: farthest-point prototype
     counts; removed from the environment before hist_encoder reads its own K, the kNN depth, which MaxSim does not use), ALPHA (0.8),
     MAXC (20), IMP_W (1), FP_ITERS (3), HIST_CAP (1000), MAX_EVENTS (0 = all; first N events per household, for smoke tests).
usage: SHARED_WORLD=1 GROUPNAMES=1 REALSTYLE_V4=1 TEST_SEEDS=100000-100049 uv run python scripts/hist_proto.py
"""
import datetime as dt
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

FP_KS = [int(k) for k in os.environ.pop("K", "3,5").split(",")]
sys.path.insert(0, str(Path(__file__).parent))
import hist_encoder as H  # noqa: E402
import hist_fast as HF  # noqa: E402
from two_tower import households  # noqa: E402

ENC = os.environ.get("ENC", "hist_knn_v1")
LAMBDAS = [float(x) for x in os.environ.get("LAMBDAS", "0,0.01,0.05").split(",")]
ALPHA, MAXC = float(os.environ.get("ALPHA", "0.8")), int(os.environ.get("MAXC", "20"))
IMP_W, FP_ITERS = float(os.environ.get("IMP_W", "1")), int(os.environ.get("FP_ITERS", "3"))
HIST_CAP, MAX_EVENTS = int(os.environ.get("HIST_CAP", "1000")), int(os.environ.get("MAX_EVENTS", "0"))
TAU = H.TAU


def _unit(X):
    return X / np.maximum(np.linalg.norm(X, axis=-1, keepdims=True), 1e-9)


def _ward(X):
    """cluster labels 0..k-1 for the rows of X (unit vectors): Ward merges up to height ALPHA, at most MAXC clusters"""
    from scipy.cluster.hierarchy import fcluster, linkage
    if len(X) == 1:
        return np.zeros(1, dtype=int)
    Z = linkage(X.astype(np.float64), method="ward")
    lab = fcluster(Z, t=ALPHA, criterion="distance")
    if lab.max() > MAXC:
        lab = fcluster(Z, t=MAXC, criterion="maxclust")
    return np.unique(lab, return_inverse=True)[1]


def _medoid(X):
    """for unit vectors argmin sum ||x_m - x_j||^2 = argmax x_m . sum x_j"""
    return X[int(np.argmax(X @ X.sum(0)))]


def _farthest(X, valid, k):
    """validity-aware farthest-point start of up to k prototypes, then FP_ITERS rounds of hard assignment and normalised means"""
    cand = np.flatnonzero(valid) if valid.any() else np.arange(len(X))
    first = cand[int(np.argmax(X[cand] @ X.mean(0)))]
    chosen = [first]
    best = X[cand] @ X[first]                      # each candidate's max cosine to the chosen points
    while len(chosen) < min(k, len(cand)):
        j = int(np.argmin(best))
        if best[j] >= 1 - 1e-6:                    # only duplicates left
            break
        chosen.append(cand[j])
        best = np.maximum(best, X[cand] @ X[cand[j]])
    P = X[chosen]
    for _ in range(FP_ITERS):
        a = np.argmax(X @ P.T, axis=1)
        P = _unit(np.stack([X[a == c].mean(0) if (a == c).any() else P[c] for c in range(len(P))]))
    return P


def _protos(X, t, valid):
    """{arm: (prototypes [p, d], bias [p])} for one category's filings X (unit, oldest first), their calendar days t and validity"""
    lab = _ward(X)
    k = lab.max() + 1
    med = np.stack([_medoid(X[lab == c]) for c in range(k)])
    cen = _unit(np.stack([X[lab == c].mean(0) for c in range(k)]))
    zero = np.zeros(k, dtype=np.float32)
    out = {"medoid": (med, zero), "centroid": (cen, zero)}
    for lam in LAMBDAS:
        z = lam * (t - t.max())                    # shifted for stability; the shift is per category, so add it back
        li = np.array([np.log(np.exp(z[lab == c]).sum()) + lam * t.max() for c in range(k)], dtype=np.float32)
        out[f"medoid+imp l={lam:g}"] = (med, IMP_W * li)
    for kk in FP_KS:
        P = _farthest(X, valid, kk)
        out[f"fp K={kk}"] = (P, np.zeros(len(P), dtype=np.float32))
    return out, k


def proto_scores(ev, E, b):
    """{arm: [N, C]} over cats (the hist_fast order), plus the mean number of Ward clusters per (category, day) clustering"""
    when = {t["id"]: dt.date.fromisoformat(t["date"]).toordinal() for t in b["transactions"]}
    cal = np.array([when[e["id"]] for e in ev], dtype=np.float64)
    cal -= cal.min()
    cats = sorted({c for e in ev for c in e["state"]} | {e["gold"] for e in ev})
    ci = {c: k for k, c in enumerate(cats)}
    g = np.array([ci[e["gold"]] for e in ev])
    valid = np.array([bool(e["payee"].strip()) for e in ev])
    day = np.array([e["day"] for e in ev])
    n, C = len(ev), len(cats)
    cache, members, dirty = {}, defaultdict(list), set()
    out, nclus = {}, []
    for d in np.unique(day):
        for c in dirty:                            # recluster only the categories that gained filings on an earlier day
            m = np.array(members[c][-HIST_CAP:]) if HIST_CAP else np.array(members[c])
            cache[c], k = _protos(E[m], cal[m], valid[m])
            nclus.append(k)
        dirty = set()
        q = np.flatnonzero(day == d)
        if cache:
            cs = sorted(cache)
            for arm in cache[cs[0]]:
                P = np.concatenate([cache[c][arm][0] for c in cs])
                bias = np.concatenate([cache[c][arm][1] for c in cs])
                starts = np.cumsum([0] + [len(cache[c][arm][0]) for c in cs[:-1]])
                S = (E[q] @ P.T) / TAU + bias
                A = out.setdefault(arm, np.full((n, C), -1.0 / TAU, dtype=np.float32))
                A[np.ix_(q, cs)] = np.maximum.reduceat(S, starts, axis=1)
        for i in q:
            members[g[i]].append(i)
            dirty.add(int(g[i]))
    return cats, out, float(np.mean(nclus)) if nclus else 0.0


def main():
    a, z = map(int, os.environ.get("TEST_SEEDS", "100000-100049").split("-"))
    d = H.ENC / ENC
    m = H._model(d)
    ranks = defaultdict(lambda: defaultdict(list))
    ks, t0 = [], time.time()
    nh = 0
    for b in households("test", range(a, z + 1)):
        nh += 1
        ev = H.events(b)
        if MAX_EVENTS:
            ev = ev[:MAX_EVENTS]
        sc = HF.scores(ev, m, d, HF.SYN_CACHE)
        E = H._embed(m, [e["text"] for e in ev], {}).astype(np.float32)
        cats, pr, k = proto_scores(ev, E, b)
        assert cats == list(sc["cats"])
        ks.append(k)
        arms = {"maxsim (all filings, top-3)": sc["5 maxsim"], **{f"proto {r}": v for r, v in pr.items()}}
        ci = {c: j for j, c in enumerate(cats)}
        for i, e in enumerate(ev):
            if e["gold"] not in e["state"]:
                continue
            kk = np.array([ci[c] for c in e["state"]])
            gi = ci[e["gold"]]
            for r, A in arms.items():
                rk = int((A[i, kk] >= A[i, gi]).sum())  # ties count against the gold (the gold itself is one of them)
                for grp in ("all",) + (() if e["seen"] else ("first-time",)):
                    ranks[r][grp].append(rk)
        print(f"household {b.get('name', nh)}: {len(ev)} events, {k:.1f} Ward clusters per clustering, {time.time() - t0:.0f}s", flush=True)
    print(f"\n**{ENC}, {nh} held-out households: % top-1 / top-3 / top-10** (ALPHA {ALPHA:g}, MAXC {MAXC}, IMP_W {IMP_W:g}, "
          f"FP_ITERS {FP_ITERS}, HIST_CAP {HIST_CAP}; {np.mean(ks):.1f} Ward clusters per category clustering)\n")
    print("| reader | all | first-time payee |\n|---|---|---|")
    for r, st in ranks.items():
        cells = [" / ".join(f"{100 * (np.array(st[g]) <= k).mean():.1f}" for k in (1, 3, 10)) + f" (n={len(st[g])})" for g in ("all", "first-time")]
        print(f"| {r} | " + " | ".join(cells) + " |", flush=True)


if __name__ == "__main__":
    main()
