"""Real-use replicate (PLAN step 22: REAL-1, REAL-4, GRAPH-2, GRAPH-6) on the frozen transaction history of
`ai_experiments.transactions` (240 merchants, half real chains and half opaque, Zipf frequencies, noisy strings, amounts,
weekdays, imbalanced categories), with frozen encoders: no training, so the whole grid runs in minutes.

Protocol (REAL-4): a trial draws k labelled transactions per category from the history (k in 1, 3, 10; a category with fewer
gets all it has) and classifies the rest, 12-way. Classifiers, all on unit-normalised embeddings of the statement string
(raw or through merchants.normalize):
  proto      nearest mean of the k labelled embeddings per category (section 6.3 / 24)
  proto+af   the same with the amount band (7 one-hot) and weekday (7 one-hot) appended to the text embedding, weighted FEAT_W
  lp         label propagation over a kNN graph of ALL transaction embeddings, labelled and not (Zhou et al. 2003: F = (I - a S)^-1 Y,
             S the symmetric-normalised kNN affinity, a = 0.9, k = 10 neighbours)  (GRAPH-2)
  name       the category vector is the embedding of the category's name (zero labelled examples needed)  (GRAPH-6)
  mix        the unit mean of the name vector and the k-example centroid  (GRAPH-6)
Accuracy is reported overall and by merchant frequency bucket (head / torso / tail), known vs opaque merchant, and whether the
test merchant appears among the labelled examples (seen) or not (unseen), over TRIALS trials.

usage: uv run python scripts/exp_realuse.py [minilm|bge|both]        SMOKE=1: 2 trials, k in (1, 3)
outputs: results/realuse.json, tracker experiment "realuse"
"""
import json
import os
import random
import sys
import time
from collections import defaultdict

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from ai_experiments import merchants as M
from ai_experiments import transactions as T
from ai_experiments.evals.tracker import Run
from ai_experiments.paths import ROOT

ENCODERS = {"minilm": "sentence-transformers/all-MiniLM-L6-v2", "bge": "BAAI/bge-base-en-v1.5"}
WHICH = sys.argv[1] if len(sys.argv) > 1 else "both"
SMOKE = bool(os.environ.get("SMOKE"))
TRIALS = 2 if SMOKE else 10
KS = (1, 3) if SMOKE else (1, 3, 10)
FEAT_W, LP_K, LP_ALPHA = 0.5, 10, 0.9
OUT = ROOT / "results" / f"realuse{'_smoke' if SMOKE else ''}.json"

if not T.PATH.exists():
    T.freeze()
DOC = T.load()
TX = DOC["transactions"]
CATS = M.CATEGORY_LIST
CAT_IDX = {c: i for i, c in enumerate(CATS)}
y_all = np.array([CAT_IDX[t["category"]] for t in TX])


def encode(model, texts):
    return model.encode(texts, batch_size=256, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False).astype(np.float32)


def features(E, with_af):
    if not with_af:
        return E
    af = np.zeros((len(TX), len(T.AMOUNT_BANDS) - 1 + len(T.WEEKDAYS)), dtype=np.float32)
    for i, t in enumerate(TX):
        af[i, T.amount_band(t["amount"])] = 1.0
        af[i, len(T.AMOUNT_BANDS) - 1 + T.WEEKDAYS.index(t["weekday"])] = 1.0
    X = np.concatenate([E, FEAT_W * af], 1)
    return X / np.linalg.norm(X, axis=1, keepdims=True)


def lp_predict(X, lab_idx, y_lab, n_cls):
    """Label propagation over a symmetric kNN graph on all rows; returns the argmax class for every row."""
    S = X @ X.T
    np.fill_diagonal(S, -1)
    nn = np.argpartition(-S, LP_K, axis=1)[:, :LP_K]
    W = np.zeros_like(S)
    rows = np.repeat(np.arange(len(X)), LP_K)
    W[rows, nn.ravel()] = np.maximum(S[rows, nn.ravel()], 0)
    W = np.maximum(W, W.T)
    d = W.sum(1) + 1e-8
    Sn = W / np.sqrt(d)[:, None] / np.sqrt(d)[None, :]
    Y = np.zeros((len(X), n_cls), dtype=np.float32)
    Y[lab_idx, y_lab] = 1.0
    F = np.linalg.solve(np.eye(len(X), dtype=np.float32) - LP_ALPHA * Sn, Y)
    return F.argmax(1)


def trial(rng, X, name_vecs, k, methods):
    """One draw of k labelled transactions per category; returns {method: predictions for the test rows}, test row indices."""
    by_cat = defaultdict(list)
    for i, t in enumerate(TX):
        by_cat[CAT_IDX[t["category"]]].append(i)
    lab = []
    for c, idx in by_cat.items():
        lab += rng.sample(idx, min(k, len(idx)))
    lab_set = set(lab)
    test = [i for i in range(len(TX)) if i not in lab_set]
    lab_idx = np.array(lab); y_lab = y_all[lab_idx]
    protos = np.stack([X[lab_idx[y_lab == c]].mean(0) if (y_lab == c).any() else np.zeros(X.shape[1], dtype=np.float32) for c in range(len(CATS))])
    protos /= np.linalg.norm(protos, axis=1, keepdims=True) + 1e-8
    preds = {}
    if "proto" in methods:
        preds["proto"] = (X[test] @ protos.T).argmax(1)
    if "name" in methods:
        preds["name"] = (X[test][:, :name_vecs.shape[1]] @ name_vecs.T).argmax(1)
    if "mix" in methods:
        mix = protos[:, :name_vecs.shape[1]] + name_vecs
        mix /= np.linalg.norm(mix, axis=1, keepdims=True)
        preds["mix"] = (X[test][:, :name_vecs.shape[1]] @ mix.T).argmax(1)
    if "lp" in methods:
        preds["lp"] = lp_predict(X, lab_idx, y_lab, len(CATS))[test]
    seen_merchants = {TX[i]["merchant"] for i in lab}
    return preds, test, seen_merchants


def breakdown(pred, test, seen_merchants):
    gold = y_all[test]
    groups = {"all": np.ones(len(test), bool)}
    for b in ("head", "torso", "tail"):
        groups[b] = np.array([TX[i]["bucket"] == b for i in test])
    groups["known"] = np.array([TX[i]["known"] for i in test]); groups["opaque"] = ~groups["known"]
    groups["seen_merchant"] = np.array([TX[i]["merchant"] in seen_merchants for i in test]); groups["unseen_merchant"] = ~groups["seen_merchant"]
    return {g: (float((pred[m] == gold[m]).mean()) if m.any() else None) for g, m in groups.items()}


results = json.loads(OUT.read_text()) if OUT.exists() and not SMOKE else {}
cfg = dict(trials=TRIALS, ks=list(KS), feat_w=FEAT_W, lp_k=LP_K, lp_alpha=LP_ALPHA, n_transactions=len(TX), n_merchants=DOC["n_merchants"], zipf_s=DOC["zipf_s"], version=DOC["version"])
with Run("realuse", model=",".join(ENCODERS[e] for e in (ENCODERS if WHICH == "both" else [WHICH])), config=cfg, enabled=not SMOKE) as run:
    for enc in (ENCODERS if WHICH == "both" else [WHICH]):
        t0 = time.time()
        model = SentenceTransformer(ENCODERS[enc], device="cuda")
        for norm in ("raw", "norm"):
            texts = [M.normalize(t["text"]) if norm == "norm" else t["text"] for t in TX]
            E = encode(model, texts)
            name_vecs = encode(model, CATS)
            for af in (False, True):
                X = features(E, af)
                for k in KS:
                    key = f"{enc}.{norm}.{'text+af' if af else 'text'}.k{k}"
                    methods = ("proto", "lp") + (() if af else ("name", "mix"))
                    acc = defaultdict(lambda: defaultdict(list))
                    rng = random.Random(100 + k)
                    for _ in range(TRIALS):
                        preds, test, seen = trial(rng, X, name_vecs, k, methods)
                        for meth, p in preds.items():
                            for g, v in breakdown(p, test, seen).items():
                                if v is not None:
                                    acc[meth][g].append(v)
                    r = {f"{meth}_{g}": round(100 * float(np.mean(v)), 1) for meth, gs in acc.items() for g, v in gs.items()}
                    results[key] = r
                    run.log(r, condition=key)
                    print(f"{key:28s} " + " ".join(f"{m}={r.get(m + '_all')}" for m in methods) + f"  | proto tail={r.get('proto_tail')} unseen={r.get('proto_unseen_merchant')}", flush=True)
                    OUT.parent.mkdir(exist_ok=True); OUT.write_text(json.dumps(results, indent=2))
        # zero-example classification: the category name alone, for GRAPH-6 (k = 0)
        for norm in ("raw", "norm"):
            texts = [M.normalize(t["text"]) if norm == "norm" else t["text"] for t in TX]
            E = encode(model, texts); name_vecs = encode(model, CATS)
            pred = (E @ name_vecs.T).argmax(1)
            r = {f"name_{g}": round(100 * v, 1) for g, v in breakdown(pred, list(range(len(TX))), set()).items() if v is not None}
            results[f"{enc}.{norm}.text.k0"] = r; run.log(r, condition=f"{enc}.{norm}.text.k0")
            print(f"{enc}.{norm}.text.k0 name-only: {r['name_all']}", flush=True)
        results[f"{enc}.minutes"] = round((time.time() - t0) / 60, 1)
        del model; torch.cuda.empty_cache()
    OUT.write_text(json.dumps(results, indent=2)); run.artifact(OUT)
print(f"=== realuse -> {OUT}")
