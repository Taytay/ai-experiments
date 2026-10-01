"""Tables for PLAN step 132 (owner, 2026-09-30: score a categoriser by where the right category lands among at most five suggestions,
0 for first place, 1 per place down, 10 when not shown, with a penalty for suggesting categories that make no sense, and suggesting only
categories the model has some confidence in). `scorecard.rank_effort`: right at rank r costs r - 1, not shown 10, each wrong category shown
LAM, a wrong auto-file W; the list and the auto-file decision from calibrated probabilities by expected cost. blind_v1; decider-4B (the
other-users line), the pair (x encoder), pair -> untrained 35B, all three. Also the plain ranking measures: top-1, recall@5, MRR@5.
usage: uv run python scripts/rank_tables.py
"""
import glob
import json
import sys
import warnings
from pathlib import Path

import numpy as np

from ai_experiments import scorecard as S
from ai_experiments.paths import PROCESSED

sys.path.insert(0, str(Path(__file__).parent))
from decision_tables import fmt, folds_calibrate, load, product  # noqa: E402
from ensemble_tables import DEC, ENC  # noqa: E402

BIG = "results/per_item/real6_dm_decider_decider-35b-a3b_blind_v1.noctx.jsonl"


def score(P, items, W, lam, mode="rule", confirm=0.0):
    rows = []
    for i, p in P.items():
        a = None if mode == "rule" else (False if mode == "never" else None)
        c, au, shown = S.rank_effort(p, items[i]["answer"], W, lam=lam, auto=a, confirm=confirm)
        rows.append((c, au, int(np.argmax(p)) == items[i]["answer"], len(shown), items[i]["answer"] in shown))
    v = np.array(rows, float)
    a = v[:, 1] > 0
    return dict(cost=v[:, 0].mean(), auto=100 * a.mean(), prec=100 * v[a, 2].mean() if a.any() else float("nan"),
                k=v[~a, 3].mean() if (~a).any() else float("nan"), found=100 * v[~a, 4].mean() if (~a).any() else float("nan"))


def ranking(P, items):
    r = [int(np.where(np.argsort(-p) == items[i]["answer"])[0][0]) + 1 for i, p in P.items()]
    r = np.array(r); return 100 * np.mean(r == 1), 100 * np.mean(r <= 5), np.mean(np.where(r <= 5, 1 / r, 0))


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    items = {i["id"]: i for i in json.loads((PROCESSED / "blind_v1.json").read_text())["items"]}
    encs = [load(f) for f in sorted(glob.glob(ENC))]; decs = [load(f) for f in sorted(glob.glob(DEC))]; big = load(BIG)
    systems = {"decider-4B (+ other users' line)": [folds_calibrate(d, items) for d in decs],
               "encoder x decider": [product([e, d], items) for e in encs for d in decs],
               "encoder x decider x 35B (all three)": [product([e, d, big], items) for e in encs for d in decs]}
    print("**Table RK.1: ranking quality (mean over seeds / pairs)**\n")
    print("| system | top-1 % | right in top 5 % | MRR@5 |"); print("|---|---|---|---|")
    for n, Ps in systems.items():
        v = np.array([ranking(P, items) for P in Ps]); print(f"| {n} | {v[:, 0].mean():.1f} | {v[:, 1].mean():.1f} | {v[:, 2].mean():.3f} |")
    for W in (10, 20):
        print(f"\n**Table RK.2: rank effort at W = {W} (right at rank r costs r - 1, not shown 10, each wrong category shown LAM, wrong auto-file W); no model = 10 per transaction**\n")
        print("| system | LAM | effort | auto-filed % / precision % | suggestions shown (when not auto-filed) | right one among them % | effort, never auto-file |")
        print("|---|---|---|---|---|---|---|")
        for n, Ps in systems.items():
            for lam in (0.5, 1.0, 2.0):
                r = [score(P, items, W, lam) for P in Ps]; nv = [score(P, items, W, lam, "never")["cost"] for P in Ps]
                g = lambda k: [x[k] for x in r]  # noqa: E731
                print(f"| {n} | {lam} | {fmt(g('cost'), 3)} | {fmt(g('auto'))} / {fmt(g('prec'))} | {fmt(g('k'), 2)} | {fmt(g('found'))} | {fmt(nv, 3)} |")

    print("\n**Table RK.3: with a confirm cost (accepting a shown suggestion costs CONFIRM over an auto-file), LAM = 1**\n")
    print("| system | W | confirm | effort | auto-filed % / precision % | suggestions shown | right one among them % |"); print("|---|---|---|---|---|---|---|")
    for n, Ps in systems.items():
        for W, cf in ((20, 0.5), (20, 1.0), (30, 0.5), (50, 1.0)):
            r = [score(P, items, W, 1.0, confirm=cf) for P in Ps]; g = lambda k: [x[k] for x in r]  # noqa: E731
            print(f"| {n} | {W} | {cf} | {fmt(g('cost'), 3)} | {fmt(g('auto'))} / {fmt(g('prec'))} | {fmt(g('k'), 2)} | {fmt(g('found'))} |")
