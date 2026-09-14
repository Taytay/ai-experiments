"""Scoring rules applied to saved per-item records (EVAL-1, EVAL-2, EVAL-6). No torch, no model.

A record (see `ai_experiments.scoring`) holds every option's log-probs under several premises.
Each rule here maps a record to one score per option; the prediction is the argmax. All of them
run over `results/per_item/*.jsonl` in well under a second per file, so trying a new rule never
means loading an adapter again.

  mean     sum_lp / n_tok          mean per-token log-prob: the rule every number so far used
                                   (Holtzman et al.'s AVG; OLMES "CF with token normalisation")
  sum      sum_lp                  raw sequence log-prob; favours short options
  bytes    sum_lp / n_bytes        per-byte normalisation (OLMES character normalisation)
  pmi_dc   sum_lp - dc_lp          PMI with the cue line ("Answer:" / "Label:") as domain premise:
                                   removes each option's prior (Holtzman et al. 2021; OLMES "pmi")
  bayes    sum_lp - b * n_tok      length correction with b fitted within-item per level from the
                                   records themselves (2607.12767); no content prior removed
  hybrid   hyb_lp                  option text scored after the choices are listed in the prompt
                                   (2607.12767, 2402.01781): the first tokens fix the option
  mcf      mcf_lp                  the option's letter after the listed choices (symbol scoring,
                                   OLMES MCF); small base models are often at chance here
  unc      dc_lp                   DIAGNOSTIC, not a scorer: question-free score (cue premise only).
                                   A level where this beats chance has an answerable-without-the-
                                   question artifact (Holtzman's UNC; Balepur's "choices only")

`rstd` (Zheng et al. 2023) is the standard deviation of per-option recall, a label/position-bias
measure; `histogram` counts predictions per option index (EVAL-2's constant-predictor check).
"""
from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Callable

Record = dict
Params = dict


def read_records(path: Path) -> list[Record]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _mean(r, p):   return [s / max(n, 1) for s, n in zip(r["sum_lp"], r["n_tok"])]
def _sum(r, p):    return list(r["sum_lp"])
def _bytes(r, p):  return [s / max(b, 1) for s, b in zip(r["sum_lp"], r["n_bytes"])]
def _pmi(r, p):    return [s - d for s, d in zip(r["sum_lp"], r["dc_lp"])]
def _bayes(r, p):  b = p["bayes_b"].get(r["level"], 0.0); return [s - b * n for s, n in zip(r["sum_lp"], r["n_tok"])]
def _hybrid(r, p): return list(r["hyb_lp"])
def _mcf(r, p):    return list(r["mcf_lp"])
def _unc(r, p):    return list(r["dc_lp"])


SCORERS: dict[str, Callable[[Record, Params], list[float]]] = dict(
    mean=_mean, sum=_sum, bytes=_bytes, pmi_dc=_pmi, bayes=_bayes, hybrid=_hybrid, mcf=_mcf, unc=_unc)
REAL = ("mean", "sum", "bytes", "pmi_dc", "bayes", "hybrid", "mcf")  # rules a report could adopt
DIAGNOSTIC = ("unc",)


def fit_bayes(records: list[Record]) -> dict[str, float]:
    """Within-item OLS slope of sum_lp on n_tok, per level: b = sum((n - n_i)(s - s_i)) / sum((n - n_i)^2)
    over options within each item (item means removed). 0 where every option has the same length."""
    num, den = defaultdict(float), defaultdict(float)
    for r in records:
        s, n = r["sum_lp"], r["n_tok"]
        sm, nm = sum(s) / len(s), sum(n) / len(n)
        num[r["level"]] += sum((ni - nm) * (si - sm) for si, ni in zip(s, n))
        den[r["level"]] += sum((ni - nm) ** 2 for ni in n)
    return {lv: (num[lv] / den[lv] if den[lv] > 0 else 0.0) for lv in num}


def params_for(records: list[Record]) -> Params:
    return {"bayes_b": fit_bayes(records)}


def predict(rec: Record, scorer: str, params: Params) -> int:
    sc = SCORERS[scorer](rec, params)
    return max(range(len(sc)), key=sc.__getitem__)  # first index wins ties, as the runs did


def by_level(records: list[Record]) -> dict[str, list[Record]]:
    out = defaultdict(list)
    for r in records:
        out[r["level"]].append(r)
    return out


def accuracy(records: list[Record], scorer: str, params: Params) -> dict[str, float]:
    """Per-level accuracy in percent under one rule."""
    hit, n = Counter(), Counter()
    for r in records:
        hit[r["level"]] += predict(r, scorer, params) == r["answer"]; n[r["level"]] += 1
    return {lv: round(100 * hit[lv] / n[lv], 1) for lv in n}


def histogram(records: list[Record], scorer: str, params: Params) -> dict[str, Counter]:
    """Per level: how often each option index was predicted."""
    h = defaultdict(Counter)
    for r in records:
        h[r["level"]][predict(r, scorer, params)] += 1
    return h


def rstd(records: list[Record], scorer: str, params: Params) -> dict[str, float]:
    """Per level: standard deviation (percentage points) of recall across option indices that
    occur as the gold answer. 0 means every option is recalled equally often; large means the
    rule prefers some positions or strings (Zheng et al. 2023 use it for letter bias)."""
    out = {}
    for lv, recs in by_level(records).items():
        hit, n = Counter(), Counter()
        for r in recs:
            n[r["answer"]] += 1; hit[r["answer"]] += predict(r, scorer, params) == r["answer"]
        rec = [100 * hit[i] / n[i] for i in n]
        m = sum(rec) / len(rec)
        out[lv] = round(math.sqrt(sum((x - m) ** 2 for x in rec) / len(rec)), 1)
    return out


def chance(records: list[Record]) -> dict[str, float]:
    """Per level: mean of 1/k in percent (k varies per item on the ICL suite and by attribute on L1)."""
    acc = defaultdict(list)
    for r in records:
        acc[r["level"]].append(100 / len(r["sum_lp"]))
    return {lv: round(sum(v) / len(v), 1) for lv, v in acc.items()}


def counts(records: list[Record]) -> dict[str, int]:
    return dict(Counter(r["level"] for r in records))


def halfwidth(acc_pct: float, n: int) -> float:
    """95% binomial half-width in percentage points for an accuracy on n items (p clipped away from 0/1)."""
    p = min(max(acc_pct / 100, 0.5 / n), 1 - 0.5 / n)
    return round(196 * math.sqrt(p * (1 - p) / n), 1)
