"""Tables for PLAN step 98 (REAL-22): which slice of the user's history the prompt should hold. REAL-7's 1,200 queries under each
`build_real7_slices.POLICIES` policy (real7_v2_<policy>), all 200 users.

  98.1  per policy: the ceiling, merchant shown, and per reader top-1 (seeds averaged, [min, max]), share of the ceiling, the share at
        calibrated p >= 0.9 of other users' threshold... (raw p >= 0.9 share and precision), auto-filed at the 98% target (calibrated on other users)
  98.2  top-1 by what decides the item, for the key policies
  98.3  effort (history-or-model, wrong = 5) per policy and reader
usage: uv run python scripts/slice_tables.py
"""
import json
import sys
import warnings
from pathlib import Path

import numpy as np

from ai_experiments import real6_cells as RC
from ai_experiments import scorecard as S
from ai_experiments.paths import PROCESSED

sys.path.insert(0, str(Path(__file__).parent))
import build_real7  # noqa: E402
import build_real7_slices as B  # noqa: E402
from effort_tables import calibrated, consistent  # noqa: E402

FINAL = "decider_decider-4b_none_h100bf16st800{s}_f0_ren50_dbep50_mislead_v1_alt10s_lk10_ov10{r7}_aux100_labrand26_ev10soft_lora"
READERS = [("decider-4B untrained", ["real6_dm_decider_decider-4b_real7_v2_{p}_labrand26.noctx.jsonl"]),
           ("final recipe (row 89)", [f"real6_dm_decider_{FINAL.format(s=s, r7='')}_real7_v2_{{p}}_labrand26.noctx.jsonl" for s in ("", "s1")]),
           ("+ REAL-7 users, all_kind slices (stage 2)", [f"real6_dm_decider_{FINAL.format(s=s, r7='_r7all_kind')}_real7_v2_{{p}}_labrand26.noctx.jsonl" for s in ("", "s1")]),
           ("+ REAL-7 users, mixed slices (stage 2)", [f"real6_dm_decider_{FINAL.format(s=s, r7='_r7mix')}_real7_v2_{{p}}_labrand26.noctx.jsonl" for s in ("", "s1")])]
KEY = ["recent", "self", "all_kind", "all_kind_b48"]
WHYS = ["plain", "idiosyncratic", "moved", "new_category", "rule_amount", "rule_day", "random_split"]


def sm(z):
    z = np.asarray(z, float); e = np.exp(z - z.max()); return e / e.sum()


def fmt(v):
    return "–" if not v else f"{np.mean(v):.1f}" + (f" [{min(v):.1f}, {max(v):.1f}]" if len(v) > 1 else "")


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    sets = {p: {i["id"]: i for i in json.loads((PROCESSED / f"real7_v2_{p}.json").read_text())["items"]} for p in B.POLICIES}
    items7, _, hist_of = build_real7.generate()
    hist = {}
    for x in items7:  # the history rule (last two filings agree, whole history), as section 98 / REPORT 98
        opts = [o.strip() for o in x["options"]]
        labs = [h["filed"] for h in hist_of[x["id"]][:-1] if h["merchant"] == x["merchant"]]
        c = consistent(labs, 2); hist[x["id"]] = opts.index(c) if c in opts else None
    loaded = {}
    for lab, pats in READERS:
        for p in B.POLICIES:
            runs = [r for r in (RC.load_recs(f.format(p=p)) for f in pats) if r]
            if runs:
                loaded[(lab, p)] = runs
    print("**Table 98.1: REAL-7, 1,200 queries, by the history slice in the prompt (24 rows unless b48; all users)**\n")
    print("| policy | ceiling | merchant shown | reader | top-1 | % of ceiling | raw p >= 0.9: share (precision) | auto-filed at 98% (precision) | effort, history or model |")
    print("|---|---|---|---|---|---|---|---|---|")
    for p, its in sets.items():
        ceil = 100 * np.mean([x["best"] == x["answer"] for x in its.values()]); shown = 100 * np.mean([x["in_shots"] for x in its.values()])
        for lab, _ in READERS:
            runs = loaded.get((lab, p))
            if not runs:
                continue
            t1 = [100 * np.mean([int(np.argmax(r[i]["sum_lp"])) == its[i]["answer"] for i in r]) for r in runs]
            conf = [[(sm(r[i]["sum_lp"]).max() >= 0.9, int(np.argmax(r[i]["sum_lp"])) == its[i]["answer"]) for i in r] for r in runs]
            sh = np.mean([100 * np.mean([c for c, _ in cs]) for cs in conf]); pr = np.mean([100 * np.mean([o for c, o in cs if c]) for cs in conf])
            cov, prec, eff = [], [], []
            for r in runs:
                cal = calibrated(r, its); ids = sorted(cal)
                a = [cal[i]["a98"] for i in ids]; ok = [int(np.argmax(cal[i]["p"])) == its[i]["answer"] for i in ids]
                cov.append(100 * np.mean(a)); prec.append(100 * np.mean([o for x, o in zip(a, ok) if x]) if any(a) else float("nan"))
                pick = [hist[i] if hist[i] is not None else (int(np.argmax(cal[i]["p"])) if cal[i]["a98"] else None) for i in ids]
                t3 = [its[i]["answer"] in np.argsort(-cal[i]["p"])[:3] for i in ids]
                eff.append(S.effort([k is not None for k in pick], [k == its[i]["answer"] for k, i in zip(pick, ids)], t3))
            print(f"| {p} | {ceil:.1f} | {shown:.0f}% | {lab} | {fmt(t1)} | {100 * np.mean(t1) / ceil:.1f}% | {sh:.0f}% ({pr:.1f}) | {np.mean(cov):.1f} ({np.nanmean(prec):.1f}) | {np.mean(eff):.3f} |")
    print("\n**Table 98.2: top-1 by what decides the item (seeds averaged)**\n")
    n = {w: sum(x["why"] == w for x in sets["recent"].values()) for w in WHYS}
    print("| policy | reader | " + " | ".join(f"{w} (n={n[w]})" for w in WHYS) + " |"); print("|---|---|" + "---|" * len(WHYS))
    for p in KEY + ["all_embrec", "all_embname", "all_kind_undated", "self_kind", "self_cover", "kind_cover"]:
        for lab, _ in READERS:
            runs = loaded.get((lab, p))
            if runs:
                its = sets[p]
                cells = [fmt([100 * np.mean([int(np.argmax(r[i]["sum_lp"])) == its[i]["answer"] for i in r if its[i]["why"] == w]) for r in runs]) for w in WHYS]
                print(f"| {p} | {lab} | " + " | ".join(cells) + " |")
