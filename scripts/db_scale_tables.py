"""Tables for REPORT.md section 58 (PLAN step 59, REAL-17): database episodes as the fact DB grows to 1,000, 5,000 and 20,000
merchants (REAL-6's 240 plus generated opaque ones), at a fixed exposure (30 DB rows per merchant) and a fixed budget (800 database
episodes); H100, bf16, all-label, row 42's four held-out folds; REAL-6's DB-only merchants are the measure.
  58.1  per size and mode: rows per merchant, steps, training minutes (fold 0), REAL-6 by group, DB-only known / opaque, paired with no DB
usage: uv run python scripts/db_scale_tables.py
"""
import json
import math

from ai_experiments import real6_cells as RC
from ai_experiments.paths import ROOT

R = ROOT / "results"; CAT = "real6_categoriser_Qwen2.5-3B-Instruct"
COLS = ["all", "coined", "determined by category", "DB-only", "DB-only known", "DB-only opaque"]


def get(d, k):
    return next((x[k] for x in (d, d.get("stats", {}), d.get("train", {})) if isinstance(x, dict) and k in x), "-")


if __name__ == "__main__":
    base = RC.load_recs(f"{CAT}_none_h100bf16_f?_alllab_lora.noctx.jsonl")
    print("**Table 58.1: database episodes as the database grows (held-out users; accuracy %; last column: DB-only minus no DB on the same items [user interval])**\n")
    print("| merchants in the DB | mode | DB rows per merchant | steps | train minutes | " + " | ".join(COLS) + " | DB-only minus no DB |")
    print("|---|---|---|---|---|" + "---|" * (len(COLS) + 1))
    print("| none (section 55) | - | 0 | 200 | 2.9 | " + " | ".join(RC.row_cells(base, COLS)) + " | - |")
    for n in (240, 1000, 5000, 20000):
        for mode in ("exp", "bud"):
            if n == 240 and mode == "bud":
                continue
            d = round(30 * n / 9) if mode == "exp" else 800
            sfx = f"_alllab{'_dbx' + str(n - 240) if n > 240 else ''}_dbe{d}"
            rs = RC.load_recs(f"{CAT}_none_h100bf16_{mode}_f?{sfx}_lora.noctx.jsonl")
            st = json.load(open(R / f"categoriser_llm_none_h100bf16_{mode}_f0{sfx}.json"))
            print(f"| {n:,} | {'fixed exposure' if mode == 'exp' else 'fixed budget'} | {9 * d / n:.1f} | {math.ceil(1.4 * (2250 + d) / 16)} | {get(st, 'train_minutes')} | "
                  + " | ".join(RC.row_cells(rs, COLS)) + f" | {RC.paired(base, rs, 'DB-only')} |")
