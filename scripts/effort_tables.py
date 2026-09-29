"""Tables for PLAN step 97 (EVAL-10): user effort and the auto-file policy, CPU only, over saved per-item scores.

Policies (each auto-files some transactions and shows the top three suggestions for the rest):
  none          nothing auto-filed
  model         the model's answer when its calibrated confidence clears the 98% (or 95%) threshold fitted on the other users
                (temperature and threshold leave-fold-out, folds by user id mod 4, as the scorecard)
  history       the user's own latest filings of the merchant when the last k of them agree (k = 2, 3); on REAL-7 over the user's whole
                history before the query (what the product holds) and, as a second reading, over the 24 rows the prompt shows
  history+model the history's category, only when the model's answer is the same
  history|model the history's category when it applies, else the model at its threshold
Effort per transaction (`scorecard.effort`): auto-filed right 0, wrong 5 (also 3 and 10), top-3 suggestion 1, search 3.

  E.1  REAL-7 (final recipe and row 96's majority-only misfile reader, two seeds each): coverage, precision, effort, searches
  E.2  REAL-6 over all 20 users (the final recipe's four fold models)
  E.3  REAL-7, the model policy's and the history policy's wrong auto-files by what decides the item
usage: uv run python scripts/effort_tables.py
"""
import collections
import json
import sys
import warnings
from pathlib import Path

import numpy as np

from ai_experiments import calibration as C
from ai_experiments import real6 as R6
from ai_experiments import real6_cells as RC
from ai_experiments import scorecard as S
from ai_experiments.paths import PROCESSED

sys.path.insert(0, str(Path(__file__).parent))
import build_real7  # noqa: E402

R7 = "real6_dm_decider_decider_decider-4b_none_{t}_f0_ren50_dbep50_mislead_v1{sn}_alt10s_lk10_ov10_aux100_labrand26_ev10soft_lora_real7_v1_labrand26.noctx.jsonl"
R7_READERS = [("final recipe (row 89)", ""), ("+ majority-only misfiles (row 96)", "_sn15m")]
SEEDS = ("h100bf16st800", "h100bf16st800s1")
R6P = "real6_dm_decider_decider_decider-4b_none_h100bf16st800_f{f}_ren50_dbep50_mislead_v1_alt10s_lk10_ov10_aux100_labrand26_ev10soft_lora_real6_labrand26.noctx.jsonl"


def calibrated(recs, items):
    """Per item: probabilities after the leave-fold-out temperature, and whether the top one clears the 98% / 95% threshold."""
    ids = [i for i in recs if i in items]; y = {i: items[i]["answer"] for i in ids}; fold = {i: items[i]["user"] % 4 for i in ids}
    out = {}
    for f in range(4):
        fit = [i for i in ids if fold[i] != f]; own = [i for i in ids if fold[i] == f]
        if not fit or not own:
            continue
        t = C.fit_temperature([np.asarray(recs[i]["sum_lp"], float) for i in fit], [y[i] for i in fit])
        pf = [C.softmax(recs[i]["sum_lp"], t) for i in fit]; cf, kf, _, _ = C.summarise(pf, [y[i] for i in fit])
        th = {q: C.select_threshold(cf, kf, 1 - q) for q in (0.95, 0.98)}
        for i in own:
            p = C.softmax(recs[i]["sum_lp"], t)
            out[i] = dict(p=p, a98=th[0.98] is not None and p.max() >= th[0.98], a95=th[0.95] is not None and p.max() >= th[0.95])
    return out


def consistent(labels, k):
    """The category of the last k filings when they agree, else None."""
    return labels[-1] if len(labels) >= k and len(set(labels[-k:])) == 1 else None


def policies(items, cal, hist_cat):
    """{policy: (auto flags, auto right flags, top-3 flags)} over the items in cal; hist_cat: {(id, key): option index or None}."""
    ids = sorted(cal); out = {}
    top3 = [items[i]["answer"] in np.argsort(-cal[i]["p"])[:3] for i in ids]
    arg = {i: int(np.argmax(cal[i]["p"])) for i in ids}

    def add(name, pick):
        a = [pick(i) is not None for i in ids]
        out[name] = (a, [pick(i) == items[i]["answer"] for i in ids], top3)
    add("none", lambda i: None)
    for q in ("98", "95"):
        add(f"model at {q}%", lambda i, q=q: arg[i] if cal[i]["a" + q] else None)
    keys = sorted({k for _, k in hist_cat})
    for k in keys:
        h = lambda i, k=k: hist_cat[(i, k)]  # noqa: E731
        add(f"history ({k})", h)
        add(f"history ({k}) + model agrees", lambda i, h=h: h(i) if h(i) is not None and h(i) == arg[i] else None)
        add(f"history ({k}) or model at 98%", lambda i, h=h: h(i) if h(i) is not None else (arg[i] if cal[i]["a98"] else None))
    return out


def row(name, a, ok, t3):
    n = len(a); cov = 100 * np.mean(a); prec = 100 * np.mean([o for x, o in zip(a, ok) if x]) if any(a) else float("nan")
    wrong = 100 * np.mean([x and not o for x, o in zip(a, ok)])
    search = 100 * np.mean([not x and not t for x, t in zip(a, t3)])
    e = [S.effort(a, ok, t3, wrong=w) for w in (3, 5, 10)]
    return f"| {name} | {n} | {cov:.1f} | {prec:.1f} | {wrong:.1f} | {search:.1f} | {e[0]:.3f} | **{e[1]:.3f}** | {e[2]:.3f} |"


HEAD = ("| policy | n | auto-filed % | precision % | wrong auto-files, % of all | searches, % of all | effort (wrong = 3) | effort (wrong = 5) | effort (wrong = 10) |\n"
        "|---|---|---|---|---|---|---|---|---|")


def pooled(rows):
    """Concatenate one policy's flags over seeds."""
    return [sum((r[j] for r in rows), []) for j in range(3)]


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    items7, _, hist_of = build_real7.generate()
    assert R6.sha256(items7) == json.loads((PROCESSED / "real7_v1.json").read_text())["sha256"]
    it7 = {i["id"]: i for i in items7}
    hc = {}
    for i, x in it7.items():
        opts = [o.strip() for o in x["options"]]
        rows_ = [h for h in hist_of[i] if not h.get("query")]
        full = [h["filed"] for h in rows_ if h["merchant"] == x["merchant"]]
        shown = [h["filed"] for h in rows_[-build_real7.SHOTS:] if h["merchant"] == x["merchant"]]
        for k in (2, 3):
            for src, labs in (("whole history", full), ("prompt rows", shown)):
                c = consistent(labs, k)
                hc[(i, f"last {k}, {src}")] = opts.index(c) if c in opts else None
    print("**Table E.1: REAL-7 v1 (1,200 items, 200 new users), user effort by auto-file policy (two seeds pooled)**\n")
    err_by = {}
    for lab, sn in R7_READERS:
        print(f"*{lab}*\n"); print(HEAD)
        per_seed = [policies(it7, calibrated(RC.load_recs(R7.format(t=t, sn=sn)), it7), hc) for t in SEEDS]
        for name in per_seed[0]:
            print(row(name, *pooled([p[name] for p in per_seed])))
        print()
        if not sn:
            ids = sorted(calibrated(RC.load_recs(R7.format(t=SEEDS[0], sn=sn)), it7))
            for name in ("model at 98%", "history (last 2, whole history)", "history (last 2, whole history) + model agrees"):
                a, ok, _ = per_seed[0][name]
                err_by[name] = collections.Counter(it7[i]["why"] for i, x, o in zip(ids, a, ok) if x and not o)
                err_by[name + " (auto)"] = collections.Counter(it7[i]["why"] for i, x in zip(ids, a) if x)
    whys = ["plain", "idiosyncratic", "moved", "new_category", "rule_amount", "rule_day", "random_split"]
    print("**Table E.3: REAL-7, the final recipe (seed 0): wrong auto-files by what decides the item (wrong / auto-filed)**\n")
    print("| policy | " + " | ".join(whys) + " |"); print("|---|" + "---|" * len(whys))
    for name in ("model at 98%", "history (last 2, whole history)", "history (last 2, whole history) + model agrees"):
        print(f"| {name} | " + " | ".join(f"{err_by[name][w]} / {err_by[name + ' (auto)'][w]}" for w in whys) + " |")
    # REAL-6: histories from the set's users (not time-ordered: the user's labels for the merchant, in stored order)
    r6 = {i["id"]: i for i in R6.load()["items"]}; users6 = {u["user"]: u for u in R6.load()["users"]}
    rec = {}
    for f in range(4):
        rec.update({i: r for i, r in RC.load_recs(R6P.format(f=f, s="real6")).items() if i in r6 and r6[i]["user"] % 4 == f})
    hc6 = {}
    for i, x in r6.items():
        u = users6[x["user"]]; names = [c["name"] for c in u["categories"]]
        labs = [h["label"] for h in u["history"] if h["merchant"] == x["merchant"]]
        for k in (2, 3):
            c = consistent(labs, k); hc6[(i, f"last {k}")] = names.index(c) if c in names else None
    print("\n**Table E.2: REAL-6, all 20 users (the final recipe's four fold models), user effort by auto-file policy**\n"); print(HEAD)
    pol = policies(r6, calibrated(rec, r6), hc6)
    for name, v in pol.items():
        print(row(name, *v))
