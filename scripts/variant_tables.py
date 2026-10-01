"""Tables for PLAN steps 136, 137 (and 123): decider variants on blind_v1 with the other-users line, by the scores the owner asked for:
top-1; first-time payees right 1st and in the top five; new users; the user's own habits (idiosyncratic, changed mind, named); effort
under the per-item rule at W = 10 (REPORT 118); REAL-6 and novel names top-1. Mean over the seeds present.
usage: uv run python scripts/variant_tables.py
"""
import glob
import json
import sys
import warnings
from pathlib import Path

import numpy as np

from ai_experiments import real6 as R6
from ai_experiments.paths import PROCESSED

sys.path.insert(0, str(Path(__file__).parent))
from decision_tables import evaluate, folds_calibrate, load  # noqa: E402

R = "results/per_item/real6_dm_decider_decider_decider-"
MID = "_emp20_f0_ren50_dbep50_mislead_v1_alt10s_lk10_ov10_oth50_aux100_labrand255_laylabelled_shots"
VARIANTS = [
    ("decider-4B, other-users line (REPORT 123)", "4b_none_h100bf16st800{s}" + MID + "_ev10soft_lora_{set}_labrand255_laylabelled_shots.noctx.jsonl"),
    ("decider-0.8B, same recipe (row 136)", "0.8b_none_h100bf16st800{s}" + MID + "_ev10soft_lora_{set}_labrand255_laylabelled_shots.noctx.jsonl"),
    ("decider-4B + category descriptions, trained (row 137)", "4b_none_h100bf16st800{s}" + MID + "_desc_ev10soft_lora_{set}_labrand255_laylabelled_shots_desc.noctx.jsonl"),
    ("decider-4B (REPORT 123) read with descriptions, untrained on them", "4b_none_h100bf16st800{s}" + MID + "_ev10soft_lora_{set}_labrand255_laylabelled_shots_desc.noctx.jsonl"),
    ("decider-4B + 35B teacher (row 123)", "4b_none_h100bf16st800{s}" + MID + "_ev10soft_teach50_lora_{set}_labrand255_laylabelled_shots.noctx.jsonl"),
]


def files(tpl, st):
    return [f for f in (R + tpl.format(s=s, set=st) for s in ("", "s1", "s2")) if Path(f).exists()]


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    items = {i["id"]: i for i in json.loads((PROCESSED / "blind_v1_others.json").read_text())["items"]}
    ph = json.loads((PROCESSED / "blind_v1_payeehist.json").read_text())
    r6 = {i["id"]: i for i in R6.load("v1")["items"]}; nov = {i["id"]: i for i in json.loads((PROCESSED / "real6_v1_novel.json").read_text())["items"]}
    first = [i for i in items if not ph[i]]; newu = [i for i in items if items[i]["why"] == "short_history"]
    habit = [i for i in items if items[i]["why"] in ("idiosyncratic", "changed_mind", "named_category")]
    print("| variant | seeds | blind top-1 | first-time payees: right 1st / top 5 | new users right 1st | own habits right 1st | effort W=10 (per-item) | REAL-6 | novel names |")
    print("|---|---|---|---|---|---|---|---|---|")
    for name, tpl in VARIANTS:
        fs = files(tpl, "blind_v1_others")
        if not fs:
            continue
        rows = []
        for f in fs:
            rec = load(f)
            rank = {i: int(np.where(np.argsort(-v) == items[i]["answer"])[0][0]) + 1 for i, v in rec.items() if i in items}
            P = folds_calibrate(rec, items)
            rows.append([100 * np.mean([rank[i] == 1 for i in rank]), 100 * np.mean([rank[i] == 1 for i in first]), 100 * np.mean([rank[i] <= 5 for i in first]),
                         100 * np.mean([rank[i] == 1 for i in newu]), 100 * np.mean([rank[i] == 1 for i in habit]), evaluate(P, items, 10, "per-item")[2]])
        v = np.array(rows).mean(0)
        def t1(st, it):
            fs2 = files(tpl, st); return np.mean([100 * np.mean([int(np.argmax(x)) == it[i]["answer"] for i, x in load(f).items() if i in it]) for f in fs2]) if fs2 else float("nan")
        print(f"| {name} | {len(fs)} | {v[0]:.1f} | {v[1]:.1f} / {v[2]:.1f} | {v[3]:.1f} | {v[4]:.1f} | {v[5]:.3f} | {t1('real6', r6):.1f} | {t1('real6_v1_novel', nov):.1f} |")
