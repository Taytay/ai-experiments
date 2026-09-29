"""Tables for PLAN step 122 (TRAIN-13): does an exponential moving average of the LoRA weights (EMA=0.99) make decider-4B's confidence
steadier across seeds? The recipe with labelled rows, seeds 0 to 2, plain and EMA; blind_v1 per seed: top-1, auto-filing at the fixed 98%
cut and under the per-item rule at W = 10 (REPORT 118), effort; REAL-6 and novel-name top-1; the spread (max - min) over seeds.
usage: uv run python scripts/stability_tables.py
"""
import json
import sys
import warnings
from pathlib import Path

import numpy as np

from ai_experiments import real6 as R6
from ai_experiments.paths import PROCESSED

sys.path.insert(0, str(Path(__file__).parent))
from decision_tables import evaluate, folds_calibrate, load  # noqa: E402

REC = "ren50_dbep50_mislead_v1_alt10s_lk10_ov10"
F = "results/per_item/real6_dm_decider_decider_decider-4b_none_h100bf16st800{s}_emp20_f0_" + REC + "_aux100_labrand255_laylabelled_shots{ema}_ev10soft_lora_{set}_labrand255_laylabelled_shots.noctx.jsonl"


def top1(f, items):
    recs = load(f); return 100 * np.mean([int(np.argmax(v)) == items[i]["answer"] for i, v in recs.items() if i in items])


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    blind = {i["id"]: i for i in json.loads((PROCESSED / "blind_v1.json").read_text())["items"]}
    r6 = {i["id"]: i for i in R6.load("v1")["items"]}
    nov = {i["id"]: i for i in json.loads((PROCESSED / "real6_v1_novel.json").read_text())["items"]}
    print("**Table ST.1: decider-4B (labelled rows) per seed, plain and with an EMA of the LoRA weights (0.99)**\n")
    print("| weights | seed | blind top-1 | auto-filed at 98% cut | per-item W=10: auto-filed % | precision % | effort | REAL-6 | novel names |")
    print("|---|---|---|---|---|---|---|---|---|")
    for ema, lab in (("", "plain"), ("_ema99", "EMA 0.99")):
        rows = []
        for s in ("", "s1", "s2"):
            fb = F.format(s=s, ema=ema, set="blind_v1")
            if not Path(fb).exists():
                continue
            P = folds_calibrate(load(fb), blind)
            cut = evaluate(P, blind, 10, "fixed 98%"); pi = evaluate(P, blind, 10, "per-item")
            r = [top1(fb, blind), cut[0], pi[0], pi[1], pi[2], top1(F.format(s=s, ema=ema, set="real6"), r6), top1(F.format(s=s, ema=ema, set="real6_v1_novel"), nov)]
            rows.append(r)
            print(f"| {lab} | {s or 's0'} | " + " | ".join(f"{x:.3f}" if k == 4 else f"{x:.1f}" for k, x in enumerate(r)) + " |")
        v = np.array(rows)
        print(f"| {lab} | **spread** | " + " | ".join(f"{np.ptp(v[:, k]):.3f}" if k == 4 else f"{np.ptp(v[:, k]):.1f}" for k in range(v.shape[1])) + " |")
