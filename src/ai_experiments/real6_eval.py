"""Per-cell scoring shared by the REAL-6 scripts (exp_real6.py, exp_real6_fewshot.py): accuracy, 95% bootstrap interval, the section 11
null band and chance per cell and per aggregate (seen / unseen, name type, all), and the per-item JSONL writer.

A record is a dict with at least level ("R6_<seen|unseen>_<standard|renamed|new>"), answer, pred, correct and options (the per-item file
drops options); write_recs() puts it under results/per_item/real6_<tag>.<cond>.jsonl, where the table scripts pair arms item by item.
"""
import json
from collections import defaultdict

import numpy as np

from .paths import ROOT


def summarize(recs, n_boot=1000):
    """Per cell: accuracy, 95% bootstrap interval, null band; plus aggregates over seen/unseen and name types."""
    by = defaultdict(list)
    for r in recs:
        by[r["level"]].append(r)
    for r in recs:  # aggregates
        seen, nt = r["level"].split("_")[1], r["level"].split("_")[2]
        by[f"R6_{seen}_all"].append(r); by[f"R6_all_{nt}"].append(r); by["R6_all"].append(r)
    out = {}
    for lv, rs in sorted(by.items()):
        c = np.array([r["correct"] for r in rs], float); n = len(c)
        acc = 100 * c.mean()
        g1, g2 = np.random.default_rng(1), np.random.default_rng(2)
        boot = sorted(100 * c[g1.integers(0, n, n)].mean() for _ in range(n_boot)) if n > 1 else [acc, acc]
        preds = np.array([r["pred"] for r in rs]); golds = np.array([r["answer"] for r in rs])
        null = sorted(100 * (preds == g2.permutation(golds)).mean() for _ in range(n_boot)) if n > 1 else [acc, acc]
        out[lv] = round(float(acc), 1); out[lv + "_n"] = n
        out[lv + "_ci"] = [round(float(boot[int(0.025 * n_boot)]), 1), round(float(boot[int(0.975 * n_boot) - 1]), 1)]
        out[lv + "_null"] = [round(float(null[int(0.025 * n_boot)]), 1), round(float(null[int(0.975 * n_boot) - 1]), 1)]
        out[lv + "_chance"] = round(float(np.mean([100 / len(r["options"]) for r in rs])), 1)
    return out


def write_recs(tag, cond, recs, smoke=False):
    """results/per_item/real6_<tag>[_smoke].<cond>.jsonl, one record per line without the options list; returns the path."""
    p = ROOT / "results" / "per_item" / f"real6_{tag}{'_smoke' if smoke else ''}.{cond}.jsonl"
    p.parent.mkdir(exist_ok=True)
    p.write_text("\n".join(json.dumps({k: v for k, v in r.items() if k != "options"}) for r in recs) + "\n")
    return p
