"""PLAN step 208, E1 (reports/workflow_review_2026-10-05.md item 8): the history encoders' kNN and MaxSim readers, vectorised on the GPU,
with a per-budget score cache. Same definitions as hist_encoder2._scores (checked by `check`): only filings on earlier days are visible;
kNN = log of the summed exp((cos - 1) / TAU_KNN) over the K nearest earlier transactions per category (1e-9 floor); MaxSim = mean of
the 3 best cosines among the category's earlier filings, / TAU (-1 / TAU when the category has none). The loops in hist_encoder2 were
O(N^2) per event in Python (a sort per category per transaction); here each block of queries is one masked matmul, one top-K, and one
top-3 per category segment.
scores(ev, model) returns {"1 knn": [N, C], "5 maxsim": [N, C]} over `cats` (every category id seen in ev's states, in a fixed order),
plus a helper to turn a row into the {category id: score} dicts the other scripts read. Cache: one .npz per budget and encoder, keyed by
the budget's transactions and the encoder's weights; synthetic budgets under data/interim/hist_cache/ (gitignored), the owner's under
the private eval folder only (CACHE_DIR).
usage: SHARED_WORLD=1 GROUPNAMES=1 REALSTYLE_V4=1 uv run python scripts/hist_fast.py check [household seed]
"""
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import hist_encoder as H  # noqa: E402

from ai_experiments.paths import ROOT  # noqa: E402

BLOCK = 2048
SYN_CACHE = ROOT / "data" / "interim" / "hist_cache"


def _key(ev, enc_dir):
    h = hashlib.sha1()
    for e in ev:
        h.update(f"{e['id']}|{e['day']}|{e['text']}|{e['gold']}\n".encode())
    for f in sorted(Path(enc_dir).glob("*.safetensors")):
        h.update(f"{f.name}{f.stat().st_size}{f.stat().st_mtime_ns}".encode())
    h.update(f"{H.K}|{H.TAU_KNN}|{H.TAU}|{H.AMT_TEXT}".encode())
    return h.hexdigest()[:20]


def _colbert(ev, E, day, g, C, cb, ncand=50):
    """batched: every text's token vectors once (padded), then MaxSim of each query against its 50 candidates in chunked einsums.
    Checked on 3,000 events (row 208): same top-1 as hist_encoder2's reader on 2,858 of 2,866, 74 s -> 1.0 s; the rest are near-ties
    moved by bf16 token vectors computed in differently shaped batches (the original reader has the same sensitivity)."""
    import torch
    n = len(ev)
    vecs, masks = [], []
    with torch.no_grad():
        for a in range(0, n, 512):
            v, m = cb.vecs([e["text"] for e in ev[a:a + 512]])
            vecs.append(v.float()); masks.append(m)
    L = max(v.shape[1] for v in vecs)
    V = torch.cat([torch.nn.functional.pad(v, (0, 0, 0, L - v.shape[1])) for v in vecs])        # [n, L, 128]
    M = torch.cat([torch.nn.functional.pad(m, (0, L - m.shape[1])) for m in masks])               # [n, L]
    out = torch.full((n, C), float(np.log(1e-9)), device=E.device)
    for a in range(0, n, 256):
        S = E[a:a + 256] @ E.T
        S = S.masked_fill(day[a:a + 256, None] <= day[None, :], float("-inf"))
        cv, ci = S.topk(min(ncand, n), dim=1)                                                       # encoder 1's candidates
        q, qm = V[a:a + 256], M[a:a + 256]                                                          # [b, L, d], [b, L]
        d, dm = V[ci], M[ci]                                                                        # [b, k, L, d], [b, k, L]
        sim = torch.einsum("bqd,bksd->bkqs", q, d).float().masked_fill(~dm[:, :, None, :], -2)
        ms = (sim.max(-1).values * qm[:, None, :]).sum(-1) / qm.sum(-1, keepdim=True)              # [b, k] mean over query tokens
        ms = ms.masked_fill(~torch.isfinite(cv), float("-inf"))
        v20, i20 = ms.topk(min(H.K, ms.shape[1]), dim=1)
        w = torch.where(torch.isfinite(v20), torch.exp((v20 - 1) / H.TAU_KNN), torch.zeros_like(v20))
        votes = torch.zeros((ms.shape[0], C), device=E.device).scatter_add_(1, g[ci.gather(1, i20)], w)
        out[a:a + 256] = torch.where(votes > 0, torch.log(votes + 1e-9), out[a:a + 256])
    return out.cpu().numpy()


def scores(ev, model, enc_dir, cache_dir=None, colbert=None):
    """{"cats": [category ids], "1 knn": [N, C] float32, "5 maxsim": [N, C] float32}; cached as .npz in cache_dir when given"""
    import torch
    cache = Path(cache_dir) / f"{_key(ev, enc_dir)}.npz" if cache_dir else None
    if cache and cache.exists():
        z = np.load(cache, allow_pickle=False)
        if colbert is None or "colbert" in z:
            return {"cats": list(z["cats"]), "1 knn": z["knn"], "5 maxsim": z["maxsim"], **({"6 colbert": z["colbert"]} if "colbert" in z else {})}
    cats = sorted({c for e in ev for c in e["state"]} | {e["gold"] for e in ev})
    ci = {c: k for k, c in enumerate(cats)}
    n, C = len(ev), len(cats)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    E = torch.tensor(H._embed(model, [e["text"] for e in ev], {}), device=dev, dtype=torch.float32)
    day = torch.tensor([e["day"] for e in ev], device=dev)
    g = torch.tensor([ci[e["gold"]] for e in ev], device=dev)
    order = torch.argsort(g, stable=True)  # columns grouped by category for the per-category top-3
    gs = g[order]
    bounds = torch.searchsorted(gs, torch.arange(C + 1, device=dev))
    knn = torch.full((n, C), float(np.log(1e-9)), device=dev)
    mx = torch.full((n, C), -1.0 / H.TAU, device=dev)
    for a in range(0, n, BLOCK):
        S = E[a:a + BLOCK] @ E.T
        S = S.masked_fill(day[a:a + BLOCK, None] <= day[None, :], float("-inf"))
        v, ix = S.topk(min(H.K, n), dim=1)
        w = torch.where(torch.isfinite(v), torch.exp((v - 1) / H.TAU_KNN), torch.zeros_like(v))
        votes = torch.zeros((S.shape[0], C), device=dev).scatter_add_(1, g[ix], w)
        knn[a:a + BLOCK] = torch.where(votes > 0, torch.log(votes + 1e-9), knn[a:a + BLOCK])
        So = S[:, order]
        for c in range(C):
            lo, hi = int(bounds[c]), int(bounds[c + 1])
            if hi == lo:
                continue
            t = So[:, lo:hi].topk(min(3, hi - lo), dim=1).values
            fin = torch.isfinite(t)
            cnt = fin.sum(1)
            mean = torch.where(fin, t, torch.zeros_like(t)).sum(1) / cnt.clamp(min=1)
            mx[a:a + BLOCK, c] = torch.where(cnt > 0, mean / H.TAU, mx[a:a + BLOCK, c])
    out = {"cats": cats, "1 knn": knn.cpu().numpy(), "5 maxsim": mx.cpu().numpy()}
    if colbert is not None:  # hist_encoder2's reader 6: token MaxSim re-scores encoder 1's 50 nearest earlier transactions; the 20 best vote
        out["6 colbert"] = _colbert(ev, E, day, g, C, colbert)
    if cache:
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache, cats=np.array(cats), knn=out["1 knn"], maxsim=out["5 maxsim"], **({"colbert": out["6 colbert"]} if "6 colbert" in out else {}))
        if "ynab-real-eval" in str(cache):
            os.chmod(cache, 0o600)
    return out


def as_dicts(sc, ev):
    """the {reader: {category id: score}} per event that hist_encoder2._scores returns, over each event's visible categories"""
    ci = {c: k for k, c in enumerate(sc["cats"])}
    out = []
    for i, e in enumerate(ev):
        ks = [ci[c] for c in e["state"]]
        ids = list(e["state"])
        out.append({r: dict(zip(ids, sc[r][i, ks].tolist())) for r in ("1 knn", "5 maxsim", "6 colbert") if r in sc})
    return out


def check(seed):
    """the fast readers against hist_encoder2._scores on one synthetic household: ranks of the gold and score differences"""
    import hist_encoder2 as H2
    from two_tower import households
    b = next(households("test", [seed]))
    ev = H2.events(b)
    m1, m4, mt = H._model(H.OUT1), H._model(H2.OUT4), H._model(H.ENC / "two_tower_v1")
    mc = H2.ColBERT(H2.OUT6) if H2.OUT6.exists() else None
    if os.environ.get("CHECK_EVENTS"):  # ColBERT's slow reader is ~1 event per 15 ms: check on the first N events
        ev = ev[:int(os.environ["CHECK_EVENTS"])]
    t = time.time()
    slow = H2._scores(ev, m1, m4, mt, mc)
    ts = time.time() - t
    t = time.time()
    fast = as_dicts(scores(ev, m1, H.OUT1, colbert=mc), ev)
    tf = time.time() - t
    for r in ("1 knn", "5 maxsim") + (("6 colbert",) if mc is not None else ()):
        same_top, diff = 0, 0.0
        n = 0
        for s, f, e in zip(slow, fast, ev):
            if e["gold"] not in e["state"]:
                continue
            n += 1
            a, b2 = s[r], f[r]
            same_top += max(a, key=a.get) == max(b2, key=b2.get) or abs(max(a.values()) - max(b2.values())) < 1e-4
            diff = max(diff, max(abs(a[c] - b2[c]) for c in a))
        print(f"{r}: same top-1 on {same_top}/{n}; max |score difference| {diff:.2e}")
    print(f"{len(ev)} events: hist_encoder2._scores {ts:.0f}s, hist_fast {tf:.1f}s")


if __name__ == "__main__":
    if sys.argv[1] == "check":
        check(int(sys.argv[2]) if len(sys.argv) > 2 else 100000)
