"""Tables for REPORT.md section 54 (PLAN step 34, INFRA-1): the same run on the 3090 and on Modal's H100.

  54.1  row 56's all-label run (no DB, 200 steps, seed 0, 4-bit) on the 3090 (4 x 4 sequences per step), the H100 with the same
        micro-batches, and the H100 with one 16-sequence pass per step (MICRO=16): minutes, tokens per second, peak memory, final loss,
        REAL-6 accuracy with the user interval, and each H100 run paired with the 3090 run (share of the same predictions, difference)
usage: uv run python scripts/modal_repro_tables.py
"""
import json

from ai_experiments import real6_cells as RC
from ai_experiments.paths import ROOT

R = ROOT / "results"
CAT = "categoriser_Qwen2.5-3B-Instruct"
RUNS = [("3090, 4 x 4", R / f"categoriser_llm_none_alllab.json", f"real6_{CAT}_none_alllab_lora.noctx.jsonl"),
        ("H100, 4 x 4", R / "modal_repro_h100x4" / "categoriser_llm_none_alllab.json", f"../modal_repro_h100x4/per_item/real6_{CAT}_none_alllab_lora.noctx.jsonl"),
        ("H100, 1 x 16", R / "categoriser_llm_none_mb16_alllab.json", f"real6_{CAT}_none_mb16_alllab_lora.noctx.jsonl")]


def get(d, k):
    return next((x[k] for x in (d, d.get("stats", {}), d.get("train", {})) if isinstance(x, dict) and k in x), "-")


if __name__ == "__main__":
    print("**Table 54.1: one run on two GPUs (row 56's all-label no-DB run, 200 steps, seed 0, 4-bit; interval = users resampled; paired with the 3090 run)**\n")
    print("| GPU, micro-batches | train minutes | tokens/s | peak GiB | final loss | REAL-6 all [interval] | same predictions as the 3090 | minus the 3090 [interval] |")
    print("|---|---|---|---|---|---|---|---|")
    base = RC.load_recs(RUNS[0][2])
    for label, stats, pat in RUNS:
        d = json.load(open(stats)); r = RC.load_recs(pat); lo, hi = RC.user_ci({i: x["correct"] for i, x in r.items()})
        same = sum(r[i]["pred"] == base[i]["pred"] for i in base) / len(base) if r is not base else 1.0
        diff = RC.paired(base, r) if label != RUNS[0][0] else "-"
        print(f"| {label} | {get(d, 'train_minutes')} | {get(d, 'tok_per_s')} | {get(d, 'peak_alloc_GiB')} | {get(d, 'final_loss')} | {100 * sum(x['correct'] for x in r.values()) / len(r):.1f} [{lo}, {hi}] | {100 * same:.1f} | {diff} |")
