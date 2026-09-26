"""Tables for REPORT.md section 55 (PLAN step 58, REAL-16): database episodes against the record in the prompt with the same
information. Every arm on Modal (H100, bf16, one 16-sequence pass per step, 200 steps), row 42's four held-out folds.

  55.1  no DB (all-label), database episodes (all-label, DBEP=0.5, no record at test), the record in the prompt (products), the record
        with the category (REC_CAT); by name type, REPORT.md 48's groups, DB-only known and opaque; user interval
  55.2  paired differences [user-resampled interval]
usage: uv run python scripts/fair_record_tables.py
"""
from ai_experiments import real6_cells as RC

CAT = "real6_categoriser_Qwen2.5-3B-Instruct"
ARMS = [("no DB (all-label)", f"{CAT}_none_h100bf16_f?_alllab_lora.noctx.jsonl"), ("database episodes (all-label), no record at test", f"{CAT}_none_h100bf16_f?_alllab_dbep50_lora.noctx.jsonl"),
        ("record in the prompt, products", f"{CAT}_ret_h100bf16_f?_lora.ctx.jsonl"), ("record in the prompt, with the category", f"{CAT}_ret_h100bf16_f?_reccat_lora.ctx.jsonl")]
COLS = ["all", "standard", "renamed", "coined", "in history", "determined by category", "split category", "DB-only", "DB-only known", "DB-only opaque"]
PAIRS = [(0, 1), (0, 3), (2, 3), (3, 1)]

if __name__ == "__main__":
    R = [RC.load_recs(p) for _, p in ARMS]
    print("**Table 55.1: database episodes against the record in the prompt at equal information (held-out users; accuracy %; interval = users resampled)**\n")
    print("| arm | " + " | ".join(COLS) + " | interval (all) |")
    print("|---|" + "---|" * (len(COLS) + 1))
    for (label, _), r in zip(ARMS, R):
        lo, hi = RC.user_ci({i: x["correct"] for i, x in r.items()})
        print(f"| {label} | " + " | ".join(RC.row_cells(r, COLS)) + f" | [{lo}, {hi}] |")
    cols = ["all", "coined", "determined by category", "DB-only", "DB-only opaque"]
    print("\n**Table 55.2: paired differences on the same items, points [user-resampled 95% interval]**\n")
    print("| comparison | " + " | ".join(cols) + " |")
    print("|---|" + "---|" * len(cols))
    for a, b in PAIRS:
        print(f"| {ARMS[b][0]} minus {ARMS[a][0]} | " + " | ".join(RC.paired(R[a], R[b], c) for c in cols) + " |")
