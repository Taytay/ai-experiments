"""Uncertainty from the saved per-item records (STAT-2, EVAL-6, REPORT-2). No torch, no model.

Three things every accuracy in a report should carry:

  bootstrap_ci   percentile 95% interval from resampling items with replacement (per level)
  null_band      empirical chance: the 2.5th and 97.5th percentile of accuracy when the gold
                 indices are permuted across the items of a level (within equal option counts),
                 so a rule that always picks one option is scored against how often that option
                 is gold, not against 1/k. A reported accuracy inside the band is not evidence.
  paired         two arms on the same frozen items: per-level difference, exact McNemar test on the
                 discordant items, bootstrap CI of the paired difference, and the flip counts
                 (items one arm gets right and the other wrong)

All randomness is seeded; 1,000 draws run in a few seconds per arm.
"""
from __future__ import annotations

import math
import random
from collections import defaultdict

from .scorers import Params, Record, by_level, predict


def _hits(records: list[Record], scorer: str, params: Params) -> dict[str, list[bool]]:
    out = defaultdict(list)
    for r in records:
        out[r["level"]].append(predict(r, scorer, params) == r["answer"])
    return out


def _pct(xs: list[float], q: float) -> float:
    xs = sorted(xs)
    i = min(len(xs) - 1, max(0, int(round(q * (len(xs) - 1)))))
    return xs[i]


def bootstrap_ci(records: list[Record], scorer: str, params: Params, n_boot: int = 1000, seed: int = 0) -> dict[str, tuple[float, float]]:
    """Per level: (lo, hi) of the 95% percentile bootstrap of accuracy, in percent."""
    rng = random.Random(seed)
    out = {}
    for lv, hits in _hits(records, scorer, params).items():
        n = len(hits)
        accs = [100 * sum(rng.choice(hits) for _ in range(n)) / n for _ in range(n_boot)]
        out[lv] = (round(_pct(accs, 0.025), 1), round(_pct(accs, 0.975), 1))
    return out


def null_band(records: list[Record], scorer: str, params: Params, n_perm: int = 1000, seed: int = 0) -> dict[str, tuple[float, float, float]]:
    """Per level: (lo, hi, mean) accuracy in percent when gold indices are permuted across the
    level's items, separately within each option count k, with the predictions held fixed."""
    rng = random.Random(seed)
    out = {}
    for lv, recs in by_level(records).items():
        preds = [predict(r, scorer, params) for r in recs]
        groups = defaultdict(list)
        for i, r in enumerate(recs):
            groups[len(r["sum_lp"])].append(i)
        accs = []
        for _ in range(n_perm):
            hit = 0
            for idx in groups.values():
                golds = [recs[i]["answer"] for i in idx]
                rng.shuffle(golds)
                hit += sum(preds[i] == g for i, g in zip(idx, golds))
            accs.append(100 * hit / len(recs))
        out[lv] = (round(_pct(accs, 0.025), 1), round(_pct(accs, 0.975), 1), round(sum(accs) / len(accs), 1))
    return out


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value from the discordant counts (b: only A right, c: only B right)."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def paired(recs_a: list[Record], recs_b: list[Record], scorer: str, params_a: Params, params_b: Params,
           n_boot: int = 1000, seed: int = 0) -> dict[str, dict]:
    """Per level, arms A and B on the same item ids: acc_a, acc_b, diff (B - A) with a bootstrap CI,
    flips (only_a, only_b: items exactly one arm gets right), exact McNemar p. Items missing from
    either arm are skipped; levels with no shared items are omitted."""
    rng = random.Random(seed)
    hit_a = {r["id"]: predict(r, scorer, params_a) == r["answer"] for r in recs_a}
    hit_b = {r["id"]: predict(r, scorer, params_b) == r["answer"] for r in recs_b}
    lv_of = {r["id"]: r["level"] for r in recs_a}
    ids_by_lv = defaultdict(list)
    for i in hit_a:
        if i in hit_b:
            ids_by_lv[lv_of[i]].append(i)
    out = {}
    for lv, ids in ids_by_lv.items():
        n = len(ids)
        pairs = [(hit_a[i], hit_b[i]) for i in ids]
        only_a = sum(1 for x, y in pairs if x and not y)
        only_b = sum(1 for x, y in pairs if y and not x)
        diffs = []
        for _ in range(n_boot):
            s = [rng.choice(pairs) for _ in range(n)]
            diffs.append(100 * (sum(y for _, y in s) - sum(x for x, _ in s)) / n)
        out[lv] = dict(n=n, acc_a=round(100 * sum(x for x, _ in pairs) / n, 1), acc_b=round(100 * sum(y for _, y in pairs) / n, 1),
                       diff=round(100 * (only_b - only_a) / n, 1), ci=(round(_pct(diffs, 0.025), 1), round(_pct(diffs, 0.975), 1)),
                       only_a=only_a, only_b=only_b, p=round(mcnemar_exact(only_a, only_b), 4))
    return out
