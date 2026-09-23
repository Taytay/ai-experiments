"""Tables for PLAN step 49 (STAT-4): calibration and selective accuracy of the REAL-6 categorisers, CPU only, from the saved per-option
scores in results/per_item/real6_*.jsonl (`sum_lp`, `n_tok` per option), with `ai_experiments.calibration`.

Protocol. Scores per option under the mean-per-token rule (sum_lp / n_tok, the rule every REAL-6 table reports) and the sum rule;
probabilities softmax(z / T) over the user's categories. Leave-users-out: the users fall in four folds (user id mod 4, row 42's
split); each fold's items get the temperature fitted by NLL on the other three folds' items and the auto-apply threshold chosen on
the other three folds' tempered confidences, so no user's items fit their own temperature or threshold. For the fold adapters of
row 42 every item is also scored by an adapter that never trained on its user.

  C.1  per arm: T (mean of the four fold fits), accuracy, NLL / Brier / ECE raw and tempered, AURC, and the auto-apply operating
       points: oracle coverage at 95% and 90% precision (threshold on the same items) and the out-of-fold one (threshold from the
       other users) with its realised precision; user-resampled intervals on tempered ECE and out-of-fold coverage at 95%
  C.2  the seeded arms as mean +- sd over seeds
  C.3  cross-arm transfer: each arm tempered with the other arm's fold temperatures
  C.4  the 95% operating point by REPORT.md 48's corrected groups: accuracy, mean tempered confidence, share auto-applied, precision
       of the auto-applied
usage: [PER_ITEM=<dir>] uv run python scripts/calibration_tables.py      prints markdown (both rules; C.3 and C.4 under the mean rule)
"""
import json
import os
from pathlib import Path

import numpy as np

from ai_experiments import calibration as C
from ai_experiments import real6_cells as RC
from ai_experiments.paths import ROOT

P = Path(os.environ.get("PER_ITEM", ROOT / "results" / "per_item"))  # PER_ITEM: read another checkout's records (a chain writing there)
CAT = "categoriser_Qwen2.5-3B-Instruct"
RULES = ("mean", "sum")
# (label, [per-item stems, one per seed or one per fold when merged], condition, merged folds)
ARMS = [("instruct base, no record", ["Qwen2.5-3B-Instruct"], "noctx", False),
        ("instruct base + record", ["Qwen2.5-3B-Instruct"], "ctx", False),
        ("SFT no DB, 200 steps", [f"{CAT}_none_lora", f"{CAT}_none_s1_lora", f"{CAT}_none_s2_lora"], "noctx", False),
        ("SFT no DB, 800 steps", [f"{CAT}_none_st800_s{s}_lora" for s in range(3)], "noctx", False),
        ("SFT + record", [f"{CAT}_ret_lora", f"{CAT}_ret_s1_lora", f"{CAT}_ret_s2_lora"], "ctx", False),
        ("SFT + retrieved record", [f"{CAT}_ret_lora"], "ret1", False),
        ("SFT no DB, 800 steps, user held out (row 42)", [f"{CAT}_none_st800_f{f}_lora" for f in range(4)], "noctx", True),
        ("SFT + record, user held out (row 42)", [f"{CAT}_ret_f{f}_lora" for f in range(4)], "ctx", True),
        ("SFT no DB, held out, rename augmentation", [f"{CAT}_none_st800_f{f}_ren50_lora" for f in range(4)], "noctx", True),
        ("SFT + record, held out, rename augmentation", [f"{CAT}_ret_f{f}_ren50_lora" for f in range(4)], "ctx", True)]
TRANSFER = [("SFT no DB, 800 steps", "SFT + record"), ("SFT + record", "SFT no DB, 800 steps"),
            ("instruct base, no record", "instruct base + record"), ("instruct base + record", "instruct base, no record")]


def load(stem, cond):
    p = P / f"real6_{stem}.{cond}.jsonl"
    return [json.loads(line) for line in open(p)] if p.exists() else []


def runs(stems, cond, merged):
    """One record list per seed, or the four fold files merged into one."""
    rs = [load(s, cond) for s in stems]
    if merged:
        return [sum(rs, [])] if all(rs) else []
    return [r for r in rs if r]


def scores(r, rule):
    s = np.asarray(r["sum_lp"], float)
    return s / np.asarray(r["n_tok"], float) if rule == "mean" else s


def fold(r):
    return r["user"] % 4


def oof(recs, rule, t_recs=None):
    """Out-of-fold tempered probabilities, fold temperatures and the out-of-fold acceptance at 95% and 90% precision.
    t_recs: fit the temperatures on these records instead (cross-arm transfer; same items, another arm)."""
    t_recs = t_recs or recs
    probs, temps, accept = {}, [], {0.95: {}, 0.90: {}}
    for k in range(4):
        fit = [r for r in t_recs if fold(r) != k]
        t = C.fit_temperature([scores(r, rule) for r in fit], [r["answer"] for r in fit]); temps.append(t)
        own_fit = [r for r in recs if fold(r) != k]
        conf_f, corr_f, _, _ = C.summarise([C.softmax(scores(r, rule), t) for r in own_fit], [r["answer"] for r in own_fit])
        for r in recs:
            if fold(r) == k:
                probs[r["id"]] = C.softmax(scores(r, rule), t)
        for prec in accept:
            th = C.select_threshold(conf_f, corr_f, 1 - prec)
            for r in recs:
                if fold(r) == k:
                    accept[prec][r["id"]] = th is not None and probs[r["id"]].max() >= th
    return probs, temps, accept


def evaluate(recs, rule, t_recs=None):
    ids = [r["id"] for r in recs]; y = [r["answer"] for r in recs]; users = {r["id"]: r["user"] for r in recs}
    raw = [C.softmax(scores(r, rule)) for r in recs]
    probs, temps, accept = oof(recs, rule, t_recs)
    tem = [probs[i] for i in ids]
    rc, rk, rb, rl = C.summarise(raw, y); tc, tk, tb, tl = C.summarise(tem, y)
    out = dict(T=float(np.mean(temps)), acc=100 * tk.mean(), nll_raw=rl.mean(), nll=tl.mean(), brier_raw=rb.mean(), brier=tb.mean(),
               ece_raw=100 * C.ece(rc, rk), ece=100 * C.ece(tc, tk), aurc=100 * C.aurc(tc, tk),
               cov95_oracle=100 * C.coverage_at_precision(tc, tk, 0.95), cov90_oracle=100 * C.coverage_at_precision(tc, tk, 0.90))
    for prec, key in ((0.95, "95"), (0.90, "90")):
        a = np.array([accept[prec][i] for i in ids])
        out["cov" + key] = 100 * a.mean(); out["prec" + key] = 100 * tk[a].mean() if a.any() else float("nan")
    # user-resampled intervals for tempered ECE and out-of-fold coverage at 95%
    us = sorted(set(users.values())); idx = {u: [j for j, i in enumerate(ids) if users[i] == u] for u in us}
    a95 = np.array([accept[0.95][i] for i in ids], float); g = np.random.default_rng(0); be, bc = [], []
    for _ in range(500):
        jj = np.concatenate([idx[u] for u in g.choice(us, len(us))])
        be.append(100 * C.ece(tc[jj], tk[jj])); bc.append(100 * a95[jj].mean())
    out["ece_ci"] = np.percentile(be, [2.5, 97.5]).round(1); out["cov95_ci"] = np.percentile(bc, [2.5, 97.5]).round(1)
    out["_items"] = dict(ids=ids, conf=tc, correct=tk, accept95=a95)
    return out


RESULTS = {}


def get(label, rule):
    if (label, rule) not in RESULTS:
        arm = next(a for a in ARMS if a[0] == label)
        rs = runs(arm[1], arm[2], arm[3])
        RESULTS[(label, rule)] = [evaluate(r, rule) for r in rs]
    return RESULTS[(label, rule)]


def f1(x):
    return f"{x:.1f}"


def t1():
    for rule in RULES:
        print(f"\n**Table C.1 ({rule} rule): calibration and the auto-apply operating point, seed 0 or the merged folds (T = mean of the four "
              "leave-users-out fits; ECE in points over 10 bins; AURC in % risk; coverage = share of items auto-applied; oracle = threshold "
              "chosen on the same items, out-of-fold = threshold chosen on the other users, with the precision it realised; intervals resample users)**\n")
        print("| arm | T | accuracy | NLL raw / tempered | Brier raw / tempered | ECE raw | ECE tempered | AURC | coverage at 95%, oracle | at 95%, out-of-fold [interval] (precision) | at 90%, oracle | at 90%, out-of-fold (precision) |")
        print("|---|---|---|---|---|---|---|---|---|---|---|---|")
        for label, *_ in ARMS:
            res = get(label, rule)
            if not res:
                continue
            d = res[0]
            print(f"| {label} | {d['T']:.2f} | {f1(d['acc'])} | {d['nll_raw']:.2f} / {d['nll']:.2f} | {d['brier_raw']:.3f} / {d['brier']:.3f} | {f1(d['ece_raw'])} | "
                  f"{f1(d['ece'])} [{d['ece_ci'][0]}, {d['ece_ci'][1]}] | {f1(d['aurc'])} | {f1(d['cov95_oracle'])} | {f1(d['cov95'])} [{d['cov95_ci'][0]}, {d['cov95_ci'][1]}] "
                  f"({f1(d['prec95'])}) | {f1(d['cov90_oracle'])} | {f1(d['cov90'])} ({f1(d['prec90'])}) |")


def t2():
    print("\n**Table C.2: the seeded arms, mean +- sd over seeds 0 to 2 (mean rule)**\n")
    print("| arm | T | accuracy | ECE raw | ECE tempered | AURC | coverage at 95%, out-of-fold | its precision |")
    print("|---|---|---|---|---|---|---|---|")
    for label, stems, _, merged in ARMS:
        res = get(label, "mean")
        if len(res) < 2:
            continue
        cell = lambda k: f"{np.mean([d[k] for d in res]):.2f} +- {np.std([d[k] for d in res], ddof=1):.2f}" if k == "T" else f"{np.mean([d[k] for d in res]):.1f} +- {np.std([d[k] for d in res], ddof=1):.1f}"  # noqa: E731
        print(f"| {label} | " + " | ".join(cell(k) for k in ("T", "acc", "ece_raw", "ece", "aurc", "cov95", "prec95")) + " |")


def t3():
    print("\n**Table C.3: cross-arm transfer, seed 0, mean rule (each arm tempered by its own leave-users-out temperatures and by the other arm's, "
          "fitted on the same users; NLL and ECE tempered; out-of-fold coverage at 95% with its realised precision)**\n")
    print("| arm | temperature from | T | NLL | ECE | coverage at 95% (precision) |")
    print("|---|---|---|---|---|---|")
    for a, b in TRANSFER:
        ra = runs(*next((x[1], x[2], x[3]) for x in ARMS if x[0] == a)); rb = runs(*next((x[1], x[2], x[3]) for x in ARMS if x[0] == b))
        if not ra or not rb:
            continue
        own, other = get(a, "mean")[0], evaluate(ra[0], "mean", t_recs=rb[0])
        for src, d in ((a + " (own)", own), (b, other)):
            print(f"| {a} | {src} | {d['T']:.2f} | {d['nll']:.2f} | {f1(d['ece'])} | {f1(d['cov95'])} ({f1(d['prec95'])}) |")


def t4():
    print("\n**Table C.4: the 95% out-of-fold operating point by corrected group, seed 0 or merged folds, mean rule (accuracy; mean tempered "
          "confidence; share auto-applied; precision of the auto-applied)**\n")
    groups = RC.KINDS[:4]
    print("| arm | " + " | ".join(groups) + " |")
    print("|---|" + "---|" * len(groups))
    for label, *_ in ARMS:
        res = get(label, "mean")
        if not res:
            continue
        it = res[0]["_items"]; cells = []
        for g in groups:
            m = np.array([RC.kind(i) == g for i in it["ids"]])
            a = it["accept95"][m].astype(bool); c = it["correct"][m]
            cells.append(f"{100 * c.mean():.0f} / {100 * it['conf'][m].mean():.0f} / {100 * a.mean():.0f} / {100 * c[a].mean():.0f}" if a.any() else
                         f"{100 * c.mean():.0f} / {100 * it['conf'][m].mean():.0f} / 0 / -")
        print(f"| {label} | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    t1(); t2(); t3(); t4()
