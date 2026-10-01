"""Tables for PLAN step 121 (EVAL-13): does the auto-file threshold hold when the population changes? On blind_v1, over the saved scores
of decider-4B (the recipe with labelled rows, REPORT 113, two seeds) and the encoder (Ettin-1B, the encoder recipe, three seeds).

  TH.1  the 98% threshold as the scorecard fits it (temperature and threshold on the other users' folds, user id mod 4): coverage and
        realised precision by user group (new users; full users whose payee is / is not in the prompt; history length; scheme)
  TH.2  threshold rules, each fitted on the other folds: one global threshold (the scorecard's); one per group; a finite-sample rule
        (the lowest of 40 candidate thresholds whose Clopper-Pearson upper bound on the error, at 0.1 / 40, is at most 2%: a union
        bound, so the error guarantee holds with probability 0.9); coverage, realised precision, the worst group's precision, effort
  TH.3  transfer: the temperature and threshold fitted on another item set (REAL-6 fold 0, its novel-name copy, novel_merchants_v1)
        and applied to blind_v1 unchanged, against blind_v1's own leave-fold-out fit
usage: uv run python scripts/threshold_tables.py
"""
import glob
import json
import warnings

import numpy as np
from scipy.stats import beta

from ai_experiments import calibration as C
from ai_experiments import real6 as R6
from ai_experiments import scorecard as S
from ai_experiments.paths import PROCESSED

REC = "ren50_dbep50_mislead_v1_alt10s_lk10_ov10"
DEC = f"results/per_item/real6_dm_decider_decider_decider-4b_none_h100bf16st800{{s}}_emp20_f0_{REC}_aux100_labrand255_laylabelled_shots_ev10soft_lora_{{set}}_labrand255_laylabelled_shots.noctx.jsonl"
ENC = f"results/per_item/real6_encmask_ettin1b_st3000_h100fresh{{s}}_f0_decnone_h100fresh{{s}}_emp20_f0_{REC}_odb20_short15_kinds20_ev10soft{{set}}.noctx.jsonl"
MODELS = {"decider-4B": [DEC.replace("{s}", s) for s in ("", "s1")], "Ettin-1B": [ENC.replace("{s}", s) for s in ("", "s1", "s2")]}
TARGET, DELTA, GRID = 0.98, 0.1, 40


def path(tpl, set_name):
    if "real6_encmask" in tpl:
        return tpl.replace("{set}", "" if set_name == "real6" else "_" + set_name)
    return tpl.replace("{set}", set_name)


def load(f):
    return {r["id"]: r for r in map(json.loads, open(f))}


def items_of(set_name):
    doc = R6.load("v1") if set_name == "real6" else json.loads((PROCESSED / f"{set_name}.json").read_text())
    return {i["id"]: i for i in doc["items"]}


def group(it):
    if it.get("why") == "short_history":
        return "new user (0-10 rows)"
    return ("payee in the prompt" if it.get("in_shots", True) else "payee not in the prompt")


def length(it):
    if it.get("why") == "short_history":
        return "new user (0-10 rows)"
    h = it.get("hist_len", 1000)
    return "history < 450 rows" if h < 450 else "history 450-700 rows" if h < 700 else "history >= 700 rows"


def scheme(it):
    return it.get("scheme", "")


def finite_threshold(conf, ok):
    """The lowest grid threshold whose Clopper-Pearson upper bound on the error among accepted items is <= 1 - TARGET at DELTA / GRID."""
    grid = np.unique(np.quantile(conf, np.linspace(0.3, 0.995, GRID)))
    best = None
    for t in grid:
        a = conf >= t; n = int(a.sum()); k = int((~ok[a]).sum())
        if n == 0:
            continue
        ucb = 1.0 if k >= n else beta.ppf(1 - DELTA / GRID, k + 1, n - k)
        if ucb <= 1 - TARGET and (best is None or t < best):
            best = t
    return best


def sequential_threshold(conf, ok, start=0.7, steps=60):
    """Learn-then-Test with fixed-sequence testing: thresholds from the one accepting the top (1 - start) of items downwards; each tested
    at the full DELTA (Clopper-Pearson upper bound on the error <= 1 - TARGET); stop at the first failure; the last passing one."""
    best = None
    for q in np.linspace(start, 0.0, steps):
        t = np.quantile(conf, q); a = conf >= t; n = int(a.sum()); k = int((~ok[a]).sum())
        ucb = 1.0 if k >= n else beta.ppf(1 - DELTA, k + 1, n - k)
        if ucb > 1 - TARGET:
            break
        best = t
    return best


def fit(recs, ids, items):
    """(temperature, global threshold, finite-sample threshold, {group fn name: {group: threshold}}) from these items."""
    y = [items[i]["answer"] for i in ids]
    t = C.fit_temperature([np.asarray(recs[i]["sum_lp"], float) for i in ids], y)
    P = [C.softmax(recs[i]["sum_lp"], t) for i in ids]
    conf = np.array([p.max() for p in P]); ok = np.array([int(np.argmax(p)) == a for p, a in zip(P, y)])
    th = C.select_threshold(conf, ok, 1 - TARGET)
    per = {}
    for fn in (group, length):
        g = np.array([fn(items[i]) for i in ids]); per[fn.__name__] = {}
        for v in set(g):
            m = g == v
            per[fn.__name__][v] = C.select_threshold(conf[m], ok[m], 1 - TARGET) if m.sum() >= 20 else th
    return t, th, (finite_threshold(conf, ok), sequential_threshold(conf, ok)), per


def apply(recs, items, ids, fitted, rule):
    """Per item: (p, accepted) under the fitted rule."""
    t, th, fin, per = fitted
    out = {}
    for i in ids:
        p = C.softmax(recs[i]["sum_lp"], t)
        cut = {"global": th, "finite": fin[0], "sequential": fin[1]}.get(rule)
        if rule in ("group", "length"):
            cut = per[rule].get(globals()[rule](items[i]), th)
        out[i] = (p, cut is not None and p.max() >= cut)
    return out


def leave_fold(recs, items, rule):
    ids = [i for i in recs if i in items]; out = {}
    for f in range(4):
        fit_ids = [i for i in ids if items[i]["user"] % 4 != f]; own = [i for i in ids if items[i]["user"] % 4 == f]
        out.update(apply(recs, items, own, fit(recs, fit_ids, items), rule))
    return out


def summary(res, items, by=None):
    ids = sorted(res); a = np.array([res[i][1] for i in ids])
    ok = np.array([int(np.argmax(res[i][0])) == items[i]["answer"] for i in ids])
    t3 = [items[i]["answer"] in np.argsort(-res[i][0])[:3] for i in ids]
    cov = 100 * a.mean(); prec = 100 * ok[a].mean() if a.any() else float("nan")
    worst = float("nan")
    if by is not None:
        g = np.array([by(items[i]) for i in ids]); vals = [100 * ok[a & (g == v)].mean() for v in set(g) if (a & (g == v)).sum() >= 10]
        worst = min(vals) if vals else float("nan")
    return cov, prec, worst, S.effort(list(a), list(ok), t3)


def fmt(v, d=1):
    v = np.array([x for x in v if not np.isnan(x)])
    return "–" if not len(v) else f"{v.mean():.{d}f} [{v.min():.{d}f}, {v.max():.{d}f}]" if len(v) > 1 else f"{v.mean():.{d}f}"


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    blind = items_of("blind_v1")
    print("**Table TH.1: the scorecard's 98% threshold on blind_v1 by user group: auto-filed % / realised precision % (mean over seeds)**\n")
    print("| group | n | " + " | ".join(MODELS) + " |"); print("|---|---|" + "---|" * len(MODELS))
    per_model = {m: [leave_fold(load(path(f, "blind_v1")), blind, "global") for f in fs] for m, fs in MODELS.items()}
    for fn in (group, length, scheme):
        for v in sorted({fn(x) for x in blind.values()}):
            ids = [i for i, x in blind.items() if fn(x) == v]; cells = []
            for m, runs in per_model.items():
                cv, pr = [], []
                for res in runs:
                    a = np.array([res[i][1] for i in ids if i in res]); ok = np.array([int(np.argmax(res[i][0])) == blind[i]["answer"] for i in ids if i in res])
                    cv.append(100 * a.mean()); pr.append(100 * ok[a].mean() if a.any() else float("nan"))
                cells.append(f"{np.mean(cv):.0f} / {np.nanmean(pr):.1f}")
            print(f"| {v} | {len(ids)} | " + " | ".join(cells) + " |")
    print("\n**Table TH.2: threshold rules on blind_v1, each fitted on the other folds (mean [range] over seeds)**\n")
    print("| model | rule | auto-filed % | precision % | worst group (payee / new user) % | effort |"); print("|---|---|---|---|---|---|")
    for m, fs in MODELS.items():
        for rule, name in (("global", "one threshold (the scorecard)"), ("group", "one per group (new user; payee in / not in the prompt)"),
                           ("length", "one per history length"), ("finite", f"finite-sample, Bonferroni: CP bound <= 2% at {DELTA}/{GRID}"),
                           ("sequential", f"finite-sample, fixed sequence from the top 30%: CP bound <= 2% at {DELTA}")):
            rows = [summary(leave_fold(load(path(f, "blind_v1")), blind, rule), blind, by=group) for f in fs]
            v = np.array(rows); print(f"| {m} | {name} | {fmt(v[:, 0])} | {fmt(v[:, 1])} | {fmt(v[:, 2])} | {fmt(v[:, 3], 3)} |")
    print("\n**Table TH.3: transfer: temperature and 98% threshold fitted on another set, applied to all of blind_v1 (mean [range] over seeds)**\n")
    print("| model | fitted on | threshold rule | auto-filed % | precision % | effort |"); print("|---|---|---|---|---|---|")
    for m, fs in MODELS.items():
        for src in ("blind_v1 (other folds)", "real6", "real6_v1_novel", "novel_merchants_v1"):
            for rule in ("global", "sequential"):
                rows = []
                for f in fs:
                    recs = load(path(f, "blind_v1"))
                    if src.startswith("blind_v1"):
                        res = leave_fold(recs, blind, rule)
                    else:
                        its = items_of(src); srec = load(path(f, src)); sids = [i for i in srec if i in its]
                        res = apply(recs, blind, [i for i in recs if i in blind], fit(srec, sids, its), rule)
                    rows.append(summary(res, blind))
                v = np.array(rows); print(f"| {m} | {src} | {rule} | {fmt(v[:, 0])} | {fmt(v[:, 1])} | {fmt(v[:, 3], 3)} |")
