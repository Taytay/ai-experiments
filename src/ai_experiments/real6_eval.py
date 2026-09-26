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
    """Per cell: accuracy, 95% bootstrap interval over items (_ci) and over users (_uci), null band; plus aggregates over seen/unseen and name types."""
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
        preds = np.array([r["pred"] for r in rs]); golds = np.array([r["answer"] for r in rs])
        if n > 1:
            boot = sorted(100 * c[g1.integers(0, n, n)].mean() for _ in range(n_boot))
            null = sorted(100 * (preds == g2.permutation(golds)).mean() for _ in range(n_boot))
            ci = [boot[int(0.025 * n_boot)], boot[int(0.975 * n_boot) - 1]]; nb = [null[int(0.025 * n_boot)], null[int(0.975 * n_boot) - 1]]
        else:  # a one-item cell (smoke runs): no interval to bootstrap
            ci = nb = [acc, acc]
        out[lv] = round(float(acc), 1); out[lv + "_n"] = n
        out[lv + "_ci"] = [round(float(ci[0]), 1), round(float(ci[1]), 1)]
        out[lv + "_null"] = [round(float(nb[0]), 1), round(float(nb[1]), 1)]
        if n > 1 and all("user" in r for r in rs):  # the set samples 20 users: resample users, items staying with their user (REAL-13)
            us = sorted({r["user"] for r in rs}); by_u = [c[[r["user"] == u for r in rs]] for u in us]; g3 = np.random.default_rng(3)
            ub = sorted(100 * np.concatenate([by_u[k] for k in g3.integers(0, len(us), len(us))]).mean() for _ in range(n_boot))
            out[lv + "_uci"] = [round(float(ub[int(0.025 * n_boot)]), 1), round(float(ub[int(0.975 * n_boot) - 1]), 1)]
        out[lv + "_chance"] = round(float(np.mean([100 / len(r["options"]) for r in rs])), 1)
    return out


def write_recs(tag, cond, recs, smoke=False):
    """results/per_item/real6_<tag>[_smoke].<cond>.jsonl, one record per line without the options list; returns the path."""
    p = ROOT / "results" / "per_item" / f"real6_{tag}{'_smoke' if smoke else ''}.{cond}.jsonl"
    p.parent.mkdir(exist_ok=True)
    p.write_text("\n".join(json.dumps({k: v for k, v in r.items() if k != "options"}) for r in recs) + "\n")
    return p
