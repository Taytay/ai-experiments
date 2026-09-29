"""Tables for PLAN step 126 (EVAL-14; owner 2026-09-29: "the 98% threshold is arbitrary; a better mechanism for scoring and determining
success"). Success is the user's expected effort per transaction (scorecard.effort: auto-filed and right 0, auto-filed and wrong W, the
answer among three suggestions 1 click, otherwise a search 3). With calibrated probabilities the decision needs no threshold: auto-file a
transaction when the expected cost of doing so, (1 - p1) W, is below the expected cost of suggesting, q3 x 1 + (1 - q3) x 3 (p1 the top
probability, q3 the top three's mass) (`scorecard.decide`). W, what a wrong auto-file costs the user, is the one product number; every
table is read over a range of W. Rules compared, each fitted on the other users' folds (user id mod 4):

  fixed 98%     the scorecard's threshold so far (a 98% precision target)
  best cut      the single confidence cut that minimises effort at this W on the other folds
  per-item      the expected-cost rule on calibrated probabilities (a temperature fitted on the other folds; no threshold)

Readers on blind_v1: decider-4B (labelled rows, two seeds), Ettin-1B (three seeds), untrained decider-35B-A3B, and combinations whose
distribution is the product of the members' calibrated distributions, recalibrated with one temperature on the other folds.
DE.1 effort by W and rule; DE.2 at W = 10: auto-filed %, precision, effort; DE.3 the W each fixed precision target implies.
usage: uv run python scripts/decision_tables.py
"""
import glob
import json
import warnings

import numpy as np

from ai_experiments import calibration as C
from ai_experiments import scorecard as S
from ai_experiments.paths import PROCESSED

REC = "ren50_dbep50_mislead_v1_alt10s_lk10_ov10"
DEC = f"results/per_item/real6_dm_decider_decider_decider-4b_none_h100bf16st800*_emp20_f0_{REC}_aux100_labrand255_laylabelled_shots_ev10soft_lora_blind_v1_labrand255_laylabelled_shots.noctx.jsonl"
ENC = f"results/per_item/real6_encmask_ettin1b_st3000_h100fresh*_f0_decnone_h100fresh*_emp20_f0_{REC}_odb20_short15_kinds20_ev10soft_blind_v1.noctx.jsonl"
BIG = "results/per_item/real6_dm_decider_decider-35b-a3b_blind_v1.noctx.jsonl"
WS = (3, 5, 10, 20, 50, 100)


def load(f):
    return {r["id"]: np.asarray(r["sum_lp"], float) for r in map(json.loads, open(f))}


def folds_calibrate(logits, items):
    """Leave-fold-out temperature on raw log-scores: {id: calibrated distribution}."""
    ids = [i for i in logits if i in items]; out = {}
    for f in range(4):
        fit = [i for i in ids if items[i]["user"] % 4 != f]; own = [i for i in ids if items[i]["user"] % 4 == f]
        t = C.fit_temperature([logits[i] for i in fit], [items[i]["answer"] for i in fit])
        for i in own:
            out[i] = C.softmax(logits[i], t)
    return out


def product(members, items):
    """The product of the members' calibrated distributions as log-scores, recalibrated with one leave-fold-out temperature."""
    cals = [folds_calibrate(m, items) for m in members]; ids = set.intersection(*(set(c) for c in cals))
    return folds_calibrate({i: sum(np.log(np.clip(c[i], 1e-12, 1)) for c in cals) for i in ids}, items)


def evaluate(P, items, W, rule):
    """(auto-filed %, precision %, effort) of one rule at one W; the threshold rules fitted on the other folds."""
    ids = sorted(P); y = np.array([items[i]["answer"] for i in ids]); users = np.array([items[i]["user"] for i in ids])
    p1 = np.array([P[i].max() for i in ids]); ok = np.array([int(np.argmax(P[i])) == a for i, a in zip(ids, y)])
    t3 = np.array([a in np.argsort(-P[i])[:3] for i, a in zip(ids, y)])
    q3 = np.array([np.sort(P[i])[-3:].sum() for i in ids])
    if rule == "per-item":
        auto = S.decide(p1, q3, W)
    else:
        auto = np.zeros(len(ids), bool)
        for f in range(4):
            fit, own = users % 4 != f, users % 4 == f
            if rule == "fixed 98%":
                th = C.select_threshold(p1[fit], ok[fit], 0.02)
            else:
                cands = np.unique(np.quantile(p1[fit], np.linspace(0, 1, 201)))
                costs = [S.effort(p1[fit] >= c, ok[fit], t3[fit], wrong=W) for c in cands] + [S.effort(np.zeros(fit.sum(), bool), ok[fit], t3[fit], wrong=W)]
                k = int(np.argmin(costs)); th = cands[k] if k < len(cands) else None
            auto[own] = (p1[own] >= th) if th is not None else False
    prec = 100 * ok[auto].mean() if auto.any() else float("nan")
    return 100 * auto.mean(), prec, S.effort(auto, ok, t3, wrong=W)


def fmt(v, d=1):
    v = np.array([x for x in v if not np.isnan(x)])
    return "–" if not len(v) else f"{v.mean():.{d}f} [{v.min():.{d}f}, {v.max():.{d}f}]" if len(v) > 1 else f"{v.mean():.{d}f}"


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    items = {i["id"]: i for i in json.loads((PROCESSED / "blind_v1.json").read_text())["items"]}
    decs, encs, big = [load(f) for f in sorted(glob.glob(DEC))], [load(f) for f in sorted(glob.glob(ENC))], load(BIG)
    readers = {
        "decider-4B": [folds_calibrate(d, items) for d in decs],
        "Ettin-1B": [folds_calibrate(e, items) for e in encs],
        "decider-35B-A3B untrained": [folds_calibrate(big, items)],
        "Ettin-1B x decider-4B": [product([e, d], items) for e in encs for d in decs],
        "Ettin-1B x decider-4B x 35B": [product([e, d, big], items) for e in encs for d in decs],
    }
    rules = ("fixed 98%", "best cut", "per-item")
    print("**Table DE.1: blind_v1, effort per transaction by what a wrong auto-file costs (W) and the auto-file rule (mean [range] over seeds / pairs); no model at all: 3.0 (every transaction a search)**\n")
    print("| reader | rule | " + " | ".join(f"W = {w}" for w in WS) + " |"); print("|---|---|" + "---|" * len(WS))
    for name, Ps in readers.items():
        for rule in rules:
            print(f"| {name} | {rule} | " + " | ".join(fmt([evaluate(P, items, w, rule)[2] for P in Ps], 3) for w in WS) + " |")
    for W in (10, 50):
        print(f"\n**Table DE.2: W = {W}: auto-filed % / precision % / effort**\n")
        print("| reader | rule | auto-filed % | precision % | effort |"); print("|---|---|---|---|---|")
        for name, Ps in readers.items():
            for rule in rules:
                v = np.array([evaluate(P, items, W, rule) for P in Ps])
                print(f"| {name} | {rule} | {fmt(v[:, 0])} | {fmt(v[:, 1])} | {fmt(v[:, 2], 3)} |")
    print("\n**Table DE.3: what each W means for decider-4B under the per-item rule: the lowest confidence it auto-files at (the marginal item), and the average precision of what it auto-files**\n")
    print("| W | lowest auto-filed confidence | auto-filed % | average precision % |"); print("|---|---|---|---|")
    for W in WS:
        lows, rows = [], []
        for P in readers["decider-4B"]:
            ids = sorted(P); p1 = np.array([P[i].max() for i in ids]); q3 = np.array([np.sort(P[i])[-3:].sum() for i in ids])
            a = S.decide(p1, q3, W); lows.append(p1[a].min() if a.any() else float("nan")); rows.append(evaluate(P, items, W, "per-item"))
        v = np.array(rows); print(f"| {W} | {fmt([100 * x for x in lows])}% | {fmt(v[:, 0])} | {fmt(v[:, 1])} |")
