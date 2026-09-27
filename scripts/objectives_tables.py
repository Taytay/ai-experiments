"""Tables for PLAN step 68 (MODEL-9): which training objective makes the [MASK] encoder's confidence trustworthy. The row 53 encoder
(ModernBERT-large, one scored [MASK] per category, 1,500 steps) trained with cross-entropy, label smoothing, log + spherical score, and
cross-entropy + Brier (exp_encoder_mask.py OBJ), four folds, each fold's model read on its own held-out users and the folds pooled.

  O.1  per objective and set: top-1, NLL and ECE raw, ECE after a temperature fitted on the other three folds' items (leave-fold-out),
       AURC, coverage at a realised 98% and 95% precision (thresholds chosen leave-fold-out), and the scorecard's bits left
usage: uv run python scripts/objectives_tables.py
"""
import json
import warnings

import numpy as np

from ai_experiments import calibration as CA
from ai_experiments import real6 as R6
from ai_experiments import real6_cells as RC
from ai_experiments import scorecard as S
from ai_experiments.paths import PROCESSED

OBJS = [("cross-entropy", ""), ("label smoothing 0.1", "_objls"), ("log + spherical score", "_objlogsph"), ("cross-entropy + Brier", "_objbrier")]


def pooled(prefix, sfx, items):
    out = {}
    for f in range(4):
        for i, r in RC.load_recs(f"{prefix}_f{f}{sfx}.noctx.jsonl").items():
            if i in items and items[i]["user"] % 4 == f:
                out[i] = r
    return out


def leave_fold(recs, items, fn):
    """fn(fit_ids, own_ids) per fold -> merged dict; the fit uses the other three folds' items."""
    res = {}
    for f in range(4):
        own = [i for i in recs if items[i]["user"] % 4 == f]; fit = [i for i in recs if items[i]["user"] % 4 != f]
        if own and fit:
            res.update(fn(fit, own))
    return res


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    sets = [("REAL-6", "real6_encmask_mbert_st1500_r68", {i["id"]: i for i in R6.load()["items"]}, None)]
    doc = json.loads((PROCESSED / "poi1_v1.json").read_text())
    sets.append(("POI-1", "real6_encmask_mbert_poi1_v1_st1500_r68", {i["id"]: i for i in doc["items"]}, {u["user"]: u for u in doc["users"]}))
    print("**Table O.1: the [MASK] encoder by training objective, four folds pooled (each item read by the model that held its user out); "
          "temperature and thresholds fitted leave-fold-out; ECE in points over 10 bins, AURC in % risk**\n")
    print("| set | objective | n | top-1 | NLL raw | ECE raw | T (mean) | ECE tempered | AURC | coverage at 98% (realised precision) | coverage at 95% (precision) | bits left |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for sname, prefix, items, users in sets:
        for oname, sfx in OBJS:
            recs = pooled(prefix, sfx, items)
            if not recs:
                continue
            ids = sorted(recs); z = {i: np.asarray(recs[i]["sum_lp"], float) for i in ids}; y = {i: recs[i]["answer"] for i in ids}
            p_raw = {i: CA.softmax(z[i]) for i in ids}
            conf = np.array([p_raw[i].max() for i in ids]); corr = np.array([int(np.argmax(z[i])) == y[i] for i in ids])
            nll_raw = float(np.mean([-np.log(max(p_raw[i][y[i]], 1e-12)) for i in ids]))
            ts = {}
            def temp(fit, own):
                t = CA.fit_temperature(CA.pad([z[i] for i in fit]), np.array([y[i] for i in fit]))
                for i in own:
                    ts[i] = t
                return {i: CA.softmax(z[i], t) for i in own}
            p_t = leave_fold(recs, items, temp)
            conf_t = np.array([p_t[i].max() for i in ids])
            def cover(prec):
                acc = []
                for f in range(4):
                    own = [k for k, i in enumerate(ids) if items[i]["user"] % 4 == f]; fit = [k for k, i in enumerate(ids) if items[i]["user"] % 4 != f]
                    th = CA.select_threshold(conf_t[fit], corr[fit], 1 - prec)
                    acc += [(conf_t[k] >= th, corr[k]) for k in own] if th is not None else [(False, corr[k]) for k in own]
                a = np.array(acc); taken = a[:, 0].astype(bool)
                return 100 * taken.mean(), (100 * a[taken, 1].mean() if taken.any() else float("nan"))
            c98, p98 = cover(0.98); c95, p95 = cover(0.95)
            sc = S.scorecard(recs, items, users=users)
            print(f"| {sname} | {oname} | {len(ids)} | {100 * corr.mean():.1f} | {nll_raw:.3f} | {100 * CA.ece(conf, corr):.1f} | {np.mean(list(ts.values())):.2f} | "
                  f"{100 * CA.ece(conf_t, corr):.1f} | {100 * CA.aurc(conf_t, corr):.1f} | {c98:.1f} ({p98:.1f}) | {c95:.1f} ({p95:.1f}) | {sc['bits']:.2f} |")
