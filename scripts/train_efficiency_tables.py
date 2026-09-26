"""Tables for REPORT.md section 52 (PLAN step 56, TRAIN-11): where the categoriser's GPU hours go and what they buy.

  52.1  the no-DB categoriser on all 20 users (seed 0, 3090): the plain recipe at 200 / 400 / 800 / 1,600 steps on the 4-bit base (row 47),
        the plain recipe on bf16 at 200 and 800 steps, the all-label loss at 100 / 200 / 400 steps and on bf16; minutes, tokens per second,
        peak memory, final loss, REPORT.md 48's groups, ARC-Easy / MMLU / symbol-label ICL
  52.2  held-out users (row 42's folds, 3090): all-label at 200 steps against the plain recipe at 800, paired, with DB-only known / opaque
  52.3  held-out users (H100, one 16-sequence pass per step): all-label with and without the final answer weighted to half of each
        sequence's loss (ANS_WEIGHT=0.5)
  52.4  all 20 users (H100): all-label at 200 steps, seeds 0-2, on 4-bit and bf16, with the general measures
usage: uv run python scripts/train_efficiency_tables.py
"""
import glob
import json

import numpy as np

from ai_experiments import real6_cells as RC
from ai_experiments.paths import ROOT

R = ROOT / "results"
CAT = "categoriser_Qwen2.5-3B-Instruct"
COLS = ["all", "in history", "determined by category", "DB-only"]


def stats(name):
    p = R / f"categoriser_llm_{name}.json"
    d = json.load(open(p)) if p.exists() else {}
    return lambda k: next((x[k] for x in (d, d.get("stats", {}), d.get("train", {})) if isinstance(x, dict) and k in x), "-")


def general(adapter):
    ps = sorted(glob.glob(str(R / f"items2_{CAT}_{adapter}_lora*.json")))  # exp_items_v2 appends the job's RUN_TAG to its name
    if not ps:
        return ["-"] * 3
    d = json.load(open(ps[0])); r = d.get("results", d)
    flat = {k: v for c in (r.values() if isinstance(next(iter(r.values())), dict) else [r]) for k, v in c.items()}
    return [f"{flat[k]:.1f}" for k in ("K_arc_easy", "K_mmlu", "ICL_symbol_mean")]


def t1():
    print("**Table 52.1: the no-DB categoriser on all 20 users, seed 0, 3090 (4 x 4 sequences per step; accuracy %; ARC-Easy, MMLU and symbol-label ICL from exp_items_v2)**\n")
    rows = [("plain, 4-bit", 200, "none"), ("plain, 4-bit", 400, "none_st400_s0"), ("plain, 4-bit", 800, "none_st800_s0"), ("plain, 4-bit", 1600, "none_st1600_s0"),
            ("plain, bf16", 200, "none_bf16"), ("plain, bf16", 800, "none_st800_bf16_s0"),
            ("all-label, 4-bit", 100, "none_st100_alllab"), ("all-label, 4-bit", 200, "none_alllab"), ("all-label, 4-bit", 400, "none_st400_alllab"),
            ("all-label, bf16", 200, "none_bf16_alllab")]
    print("| recipe | steps | minutes | tokens/s | peak GiB | final loss | " + " | ".join(COLS) + " | ARC-Easy | MMLU | ICL symbol |")
    print("|---|---|---|---|---|---|" + "---|" * (len(COLS) + 3))
    for label, steps, name in rows:
        s = stats(name); rs = RC.load_recs(f"real6_{CAT}_{name}_lora.noctx.jsonl")
        print(f"| {label} | {steps} | {s('train_minutes')} | {s('tok_per_s')} | {s('peak_alloc_GiB')} | {s('final_loss')} | " + " | ".join(RC.row_cells(rs, COLS)) + " | " + " | ".join(general(name)) + " |")


def t2():
    print("\n**Table 52.2: held-out users (row 42's four folds, 3090): all-label at 200 steps against the plain recipe at 800 steps (accuracy %; last row: all-label minus plain on the same items [user-resampled interval])**\n")
    cols = ["all", "coined", "in history", "determined by category", "DB-only known", "DB-only opaque"]
    plain = RC.load_recs(f"real6_{CAT}_none_st800_f?_lora.noctx.jsonl"); al = RC.load_recs(f"real6_{CAT}_none_f?_alllab_lora.noctx.jsonl")
    print("| recipe | " + " | ".join(cols) + " |")
    print("|---|" + "---|" * len(cols))
    print("| plain, 800 steps (77 min) | " + " | ".join(RC.row_cells(plain, cols)) + " |")
    print("| all-label, 200 steps (18 min) | " + " | ".join(RC.row_cells(al, cols)) + " |")
    print("| all-label minus plain | " + " | ".join(RC.paired(plain, al, c) for c in cols) + " |")


def t3():
    print("\n**Table 52.3: held-out users (four folds, H100, one 16-sequence pass per step, 4-bit): the answer's share of the loss**\n")
    cols = ["all", "coined", "DB-only known", "DB-only opaque"]
    base = RC.load_recs(f"real6_{CAT}_none_h100_f?_alllab_lora.noctx.jsonl"); aw = RC.load_recs(f"real6_{CAT}_none_h100_f?_alllab_aw50_lora.noctx.jsonl")
    print("| loss | " + " | ".join(cols) + " |")
    print("|---|" + "---|" * len(cols))
    print("| all-label, token mean (answer ~3 of ~85 labelled tokens) | " + " | ".join(RC.row_cells(base, cols)) + " |")
    print("| all-label, answer weighted to half of each sequence | " + " | ".join(RC.row_cells(aw, cols)) + " |")
    print("| weighted minus token mean | " + " | ".join(RC.paired(base, aw, c) for c in cols) + " |")


def t4():
    print("\n**Table 52.4: 4-bit against bf16, all-label at 200 steps on all 20 users, three seeds each (H100, one 16-sequence pass per step; mean +- sd, then the three seeds)**\n")
    print("| base | REAL-6 | ARC-Easy | MMLU | ICL symbol | training minutes |")
    print("|---|---|---|---|---|---|")
    for label, prec in (("4-bit", "h100"), ("bf16", "h100_bf16")):
        acc, gen, mins = [], [], []
        for s in range(3):
            name = f"none_{prec}_s{s}_alllab"; rs = RC.load_recs(f"real6_{CAT}_{name}_lora.noctx.jsonl")
            acc.append(100 * np.mean([r["correct"] for r in rs.values()])); gen.append([float(x) for x in general(name)]); mins.append(stats(f"none_{prec}_s{s}_alllab")("train_minutes"))
        g = np.array(gen)
        f = lambda v: f"{np.mean(v):.1f} +- {np.std(v, ddof=1):.1f} ({', '.join(f'{x:.1f}' for x in v)})"  # noqa: E731
        print(f"| {label} | {f(acc)} | {f(g[:, 0])} | {f(g[:, 1])} | {f(g[:, 2])} | {', '.join(str(m) for m in mins)} |")


if __name__ == "__main__":
    t1(); t2(); t3(); t4()
