"""Tables for PLAN steps 103 to 105: decider-4B recipes side by side on every set, seeds averaged.

  RV.1  top-1 on REAL-6 fold 0, novel names, REAL-7 v1 and v2 all_kind, blind_v1 (and its > 26-option items), blind calibrated auto-filing
        at the 98% target, the trained behaviours (misleading names in the DB, override, alternation stores / restaurants, evidence-free
        and 60 / 40 confidence)
usage: uv run python scripts/recipe_tables.py
"""
import sys
import warnings
from pathlib import Path

import numpy as np

from ai_experiments import real6_cells as RC

sys.path.insert(0, str(Path(__file__).parent))
import rematch_tables as R  # noqa: E402
from effort_tables import calibrated  # noqa: E402

A = "decider_decider-4b_none_h100bf16st{st}{s}{emp}_f0_" + R.REC + "{r7}{k}_aux100_lab{lab}_ev10soft_lora"
RECIPES = [("row 89: final recipe (rand26)", dict(emp="", r7="", lab="rand26"), ("", "s1")),
           ("rand255 + empty categories (row 103)", dict(emp="_emp20", r7="", lab="rand255"), ("", "s1", "s2")),
           ("rand255 + empty + kinds 0.1 (row 110)", dict(emp="_emp20", r7="", lab="rand255", k="_kinds10"), ("", "s1", "s2")),
           ("row 89 + 300 dated slices (row 104)", dict(emp="", r7="_r7all_kind_u2400n300", lab="rand26"), ("",)),
           ("recipe v2: rand255 + empty + 300 dated slices (row 105)", dict(emp="_emp20", r7="_r7all_kind_u2400n300", lab="rand255"), ("", "s1")),
           ("rand255 + empty, 1,600 steps (row 106)", dict(emp="_emp20", r7="", lab="rand255", st="1600"), ("", "s1")),
           ("recipe v2, 1,600 steps (row 106)", dict(emp="_emp20", r7="_r7all_kind_u2400n300", lab="rand255", st="1600"), ("", "s1"))]
for _l, _kw, _s in RECIPES:
    _kw.setdefault("st", "800"); _kw.setdefault("k", "")


def fmt(v):
    v = [x for x in v if x is not None and not np.isnan(x)]
    return "–" if not v else f"{np.mean(v):.1f}" + (f" [{min(v):.1f}, {max(v):.1f}]" if len(v) > 1 else "")


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    sets = [("", True), ("real6_v1_novel", True), ("real7_v1", False), ("real7_v2_all_kind", False), ("blind_v1", False)]
    its = {s: R.items_of(s) for s, _ in sets}; beh = {s: R.items_of(s) for s in ("mislead_v1", "override_v1", "alternation_v1")}
    print("**Table RV.1: decider-4B recipes (mean [range] over seeds)**\n")
    print("| recipe | seeds | REAL-6 | novel names | REAL-7 v1 | REAL-7 v2 | blind_v1 | blind > 26 options | blind auto-filed at 98% (precision) | misleading | override | alternation stores / restaurants | evidence-free p >= 0.9 | 60 / 40 p >= 0.9 |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for lab, kw, seeds in RECIPES:
        cols = {k: [] for k in ("s", "big", "cov", "prec", "m", "o", "a1", "a2", "ev", "sp")}; per = {s: [] for s, _ in sets}; n = 0
        for sd in seeds:
            ad = A.format(s=sd, **kw); f = lambda s: f"real6_dm_decider_{ad}_{s or 'real6'}_lab{kw['lab']}.noctx.jsonl"  # noqa: E731
            if not RC.load_recs(f("")):
                continue
            n += 1
            for s, fold0 in sets:
                r = RC.load_recs(f(s)); per[s].append(R.top1(r, its[s], fold0)[0] if r else None)
            rb = RC.load_recs(f("blind_v1")); b = its["blind_v1"]
            if rb:
                cols["big"].append(100 * np.mean([int(np.argmax(rb[i]["sum_lp"])) == b[i]["answer"] for i in rb if len(b[i]["options"]) > 26]))
                cal = calibrated(rb, b); a = [i for i in cal if cal[i]["a98"]]
                cols["cov"].append(100 * len(a) / len(cal)); cols["prec"].append(100 * np.mean([int(np.argmax(cal[i]["p"])) == b[i]["answer"] for i in a]) if a else np.nan)
            ml, ov, al = RC.load_recs(f("mislead_v1")), RC.load_recs(f("override_v1")), RC.load_recs(f("alternation_v1"))
            if ml and ov and al:
                mi, oi, ai = beh["mislead_v1"], beh["override_v1"], beh["alternation_v1"]
                ok = lambda r, it, i: int(np.argmax(r[i]["sum_lp"])) == it[i]["answer"]  # noqa: E731
                cols["m"].append(100 * np.mean([ok(ml, mi, i) for i in ml if mi[i]["mkind"] == "misleading" and mi[i]["in_db"] and mi[i]["cond"] == "none"]))
                cols["o"].append(100 * np.mean([ok(ov, oi, i) for i in ov if oi[i]["cond"] == "override"]))
                cols["a1"].append(100 * np.mean([ok(al, ai, i) for i in al if ai[i]["scen"] == "observable" and ai[i]["kind"] == "store"]))
                cols["a2"].append(100 * np.mean([ok(al, ai, i) for i in al if ai[i]["scen"] == "observable" and ai[i]["kind"] == "restaurant"]))
                cols["ev"].append(100 * np.mean([R.sm(ml[i]["sum_lp"]).max() >= 0.9 for i in ml if mi[i]["mkind"] == "neutral" and not mi[i]["in_db"] and mi[i]["cond"] == "none"]))
                cols["sp"].append(100 * np.mean([R.sm(al[i]["sum_lp"]).max() >= 0.9 for i in al if ai[i]["scen"] == "unobservable"]))
        if n:
            print(f"| {lab} | {n} | " + " | ".join(fmt(per[s]) for s, _ in sets) + f" | {fmt(cols['big'])} | {fmt(cols['cov'])} ({fmt(cols['prec'])}) | {fmt(cols['m'])} | {fmt(cols['o'])} | {fmt(cols['a1'])} / {fmt(cols['a2'])} | {fmt(cols['ev'])} | {fmt(cols['sp'])} |")
