"""PLAN step 155 (MODEL-24): (and the hybrid head trained on strands' corpus, exp_corpus_slot.py POINTER=1) do strands v19 and decider's method trained on strands' corpus (REPORT 149) complement each other on strands'
own evaluations? From the per-row log-probabilities (eval_corpus_slot.py PER_ITEM=1): each reader alone, and the product of the two
(each calibrated with a temperature fitted on the other half of the set's rows, by row parity; the product recalibrated the same way,
as decision_tables.product does by user folds); and how many of one reader's misses the other gets right. By set and task, accuracy %.
usage: uv run python scripts/corpus_ensemble.py
"""
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from ai_experiments import calibration as C

D = Path("results/per_item")
SETS = ["holdout_v5_norule", "multistep_v14_eval", "generated_v16_eval", "generated_v18_eval", "adequacy_hs2_eval", "adequacy_gen_eval"]
V19 = "corpus_strands_strands-decider-2B-hobson-v19_{s}.jsonl"
HYB = ["corpus_slot_corpusslot_qwen3.5-2b-base_h100bf16_ptr_lora_{s}.jsonl", "corpus_slot_corpusslot_qwen3.5-2b-base_h100bf16s1_ptr_lora_{s}.jsonl"]  # row 155
SLOT = ["corpus_slot_corpusslot_qwen3.5-2b-base_h100bf16_lora_{s}.jsonl", "corpus_slot_corpusslot_qwen3.5-2b-base_h100bf16s1_lora_{s}.jsonl"]


def load(f):
    return [json.loads(l) for l in open(D / f)]


def halves_calibrate(lps, labels):
    """Each half's rows tempered with the temperature fitted on the other half."""
    out = [None] * len(lps)
    for h in (0, 1):
        fit = [i for i in range(len(lps)) if i % 2 != h]
        t = C.fit_temperature([np.asarray(lps[i]) for i in fit], [labels[i] for i in fit])
        for i in range(len(lps)):
            if i % 2 == h:
                out[i] = C.softmax(np.asarray(lps[i]), t)
    return out


def acc(P, rows, sel):
    return 100 * np.mean([int(np.argmax(P[i])) == rows[i]["label"] for i in sel])


if __name__ == "__main__":
    print("| set / task | n | v19 | decider's method (2 seeds) | v19 x decider's method | hybrid head (2 seeds) | decider's misses v19 gets right | v19's misses decider gets right |")
    print("|---|---|---|---|---|---|---|---|")
    for s in SETS:
        v = load(V19.format(s=s)); labels = [r["label"] for r in v]
        Pv = halves_calibrate([r["lp"] for r in v], labels)
        slots = [load(f.format(s=s)) for f in SLOT]
        for sl in slots:
            assert [r["label"] for r in sl] == labels
        Ps = [halves_calibrate([r["lp"] for r in sl], labels) for sl in slots]
        hyb = [load(f.format(s=s)) for f in HYB if (D / f.format(s=s)).exists()]
        Ph = [[np.exp(np.asarray(r["lp"])) for r in h] for h in hyb]
        Pp = [halves_calibrate([np.log(np.clip(Pv[i], 1e-12, 1)) + np.log(np.clip(P[i], 1e-12, 1)) for i in range(len(v))], labels) for P in Ps]
        groups = defaultdict(list)
        for i, r in enumerate(v):
            groups["all"].append(i); groups[r["task"]].append(i)
        for g in ["all"] + sorted(k for k in groups if k != "all" and len(groups[k]) >= 100):
            sel = groups[g]
            okv = {i: int(np.argmax(Pv[i])) == labels[i] for i in sel}
            res = []
            for P in Ps:
                oks = {i: int(np.argmax(P[i])) == labels[i] for i in sel}
                a = [i for i in sel if not oks[i]]; b = [i for i in sel if not okv[i]]
                res.append((100 * np.mean([okv[i] for i in a]) if a else np.nan, 100 * np.mean([oks[i] for i in b]) if b else np.nan))
            print(f"| {s} / {g} | {len(sel)} | {acc(Pv, v, sel):.1f} | {np.mean([acc(P, v, sel) for P in Ps]):.1f} | "
                  f"{np.mean([acc(P, v, sel) for P in Pp]):.1f} | {np.mean([acc(P, v, sel) for P in Ph]) if Ph else float('nan'):.1f} | {np.nanmean([x[0] for x in res]):.0f}% | {np.nanmean([x[1] for x in res]):.0f}% |")
