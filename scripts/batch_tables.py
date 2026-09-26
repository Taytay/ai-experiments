"""Tables for REPORT.md section 56 (PLAN step 60, TRAIN-12): the effective batch and the pass size on the H100 (all-label, bf16,
all 20 users, one seed per setting; three batch-16 seeds from section 52.4 as the reference).
  56.1  per setting: sequences per step and per pass, steps, rate, sequences seen, minutes, tokens/s, peak memory, final loss,
        REAL-6 by group, ARC-Easy / MMLU / symbol-label ICL
usage: uv run python scripts/batch_tables.py
"""
import glob
import json

from ai_experiments import real6_cells as RC
from ai_experiments.paths import ROOT

R = ROOT / "results"; CAT = "categoriser_Qwen2.5-3B-Instruct"
RUNS = [("batch 16, seed 0", 16, 16, 200, "1e-4", "h100_bf16_s0"), ("batch 16, seed 1", 16, 16, 200, "1e-4", "h100_bf16_s1"), ("batch 16, seed 2", 16, 16, 200, "1e-4", "h100_bf16_s2"),
        ("batch 16, passes of 8", 16, 8, 200, "1e-4", "h100bf16_b16m8"), ("batch 32, equal samples", 32, 32, 100, "1.41e-4", "h100bf16_b32s100"),
        ("batch 32, equal samples, rate unscaled", 32, 32, 100, "1e-4", "h100bf16_b32s100lr1"), ("batch 64, equal samples", 64, 32, 50, "2e-4", "h100bf16_b64s50"),
        ("batch 64, equal samples, rate unscaled", 64, 32, 50, "1e-4", "h100bf16_b64s50lr1"), ("batch 32, equal steps", 32, 32, 200, "1.41e-4", "h100bf16_b32s200"),
        ("batch 64, equal steps", 64, 32, 200, "2e-4", "h100bf16_b64s200")]
COLS = ["all", "in history", "determined by category", "coined", "DB-only"]


def get(d, k):
    return next((x[k] for x in (d, d.get("stats", {}), d.get("train", {})) if isinstance(x, dict) and k in x), "-")


def general(adapter):
    ps = sorted(glob.glob(str(R / f"items2_{CAT}_{adapter}_lora*.json")))
    if not ps:
        return ["-"] * 3
    d = json.load(open(ps[0])); r = d.get("results", d)
    flat = {k: v for c in (r.values() if isinstance(next(iter(r.values())), dict) else [r]) for k, v in c.items()}
    return [f"{flat[k]:.1f}" for k in ("K_arc_easy", "K_mmlu", "ICL_symbol_mean")]


if __name__ == "__main__":
    print("**Table 56.1: batch size and pass size on the H100 (all-label no-DB categoriser, bf16, all 20 users; accuracy %)**\n")
    print("| setting | per step | per pass | steps | rate | sequences | minutes | tokens/s | peak GiB | final loss | " + " | ".join(COLS) + " | ARC-Easy | MMLU | ICL symbol |")
    print("|---|" + "---|" * (12 + len(COLS)))
    for label, b, m, st, lr, tag in RUNS:
        d = json.load(open(R / f"categoriser_llm_none_{tag}_alllab.json")); rs = RC.load_recs(f"real6_{CAT}_none_{tag}_alllab_lora.noctx.jsonl")
        print(f"| {label} | {b} | {m} | {st} | {lr} | {b * st:,} | {get(d, 'train_minutes')} | {get(d, 'tok_per_s')} | {get(d, 'peak_alloc_GiB')} | {get(d, 'final_loss')} | "
              + " | ".join(RC.row_cells(rs, COLS)) + " | " + " | ".join(general(f"none_{tag}_alllab")) + " |")
