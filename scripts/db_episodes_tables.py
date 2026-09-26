"""Tables for REPORT.md section 53 (PLAN step 57, REAL-15): the fact DB as supervised decisions. All arms on Modal (H100, 4-bit, one
16-sequence pass per step, all-label loss, 200 steps), row 42's four held-out folds, scored without the record in the prompt.

  53.1  arms: no DB, database episodes (DBEP=0.5), prose records with the category (DB=param DB_CAT=1 DB_FRAC=0.5), both
        (DB_FRAC=0.25 + DBEP=0.5); by name type, REPORT.md 48's groups, DB-only merchants known and opaque; user interval
  53.2  each arm minus no DB on the same items [user-resampled interval]
  53.3  per fold: DB-only accuracy of each arm (five users each), so one fold cannot carry the result
usage: uv run python scripts/db_episodes_tables.py
"""
import numpy as np

from ai_experiments import real6_cells as RC

CAT = "real6_categoriser_Qwen2.5-3B-Instruct"
ARMS = [("no DB", f"{CAT}_none_h100_f{{f}}_alllab_lora.noctx.jsonl"), ("database episodes", f"{CAT}_none_h100_f{{f}}_alllab_dbep50_lora.noctx.jsonl"),
        ("prose records + category", f"{CAT}_param_h100_f{{f}}_alllab_dbcat_lora.noctx.jsonl"), ("both", f"{CAT}_param_h100_f{{f}}_alllab_dbep50_dbcat_lora.noctx.jsonl")]
COLS = ["all", "standard", "renamed", "coined", "in history", "determined by category", "split category", "DB-only", "DB-only known", "DB-only opaque"]


def recs(pat, fold="?"):
    return RC.load_recs(pat.format(f=fold))


def t1():
    print("**Table 53.1: the fact DB in the weights, four held-out folds, no record in the prompt at test (accuracy %; interval = users resampled)**\n")
    print("| arm | " + " | ".join(COLS) + " | interval (all) |")
    print("|---|" + "---|" * (len(COLS) + 1))
    for label, pat in ARMS:
        r = recs(pat); lo, hi = RC.user_ci({i: x["correct"] for i, x in r.items()})
        print(f"| {label} | " + " | ".join(RC.row_cells(r, COLS)) + f" | [{lo}, {hi}] |")


def t2():
    cols = ["all", "coined", "determined by category", "DB-only", "DB-only known", "DB-only opaque"]
    print("\n**Table 53.2: each arm minus no DB on the same items, points [user-resampled 95% interval]**\n")
    print("| arm | " + " | ".join(cols) + " |")
    print("|---|" + "---|" * len(cols))
    base = recs(ARMS[0][1])
    for label, pat in ARMS[1:]:
        r = recs(pat)
        print(f"| {label} | " + " | ".join(RC.paired(base, r, c) for c in cols) + " |")


def t3():
    print("\n**Table 53.3: DB-only merchants per fold (accuracy %, items in the fold)**\n")
    print("| arm | " + " | ".join(f"fold {f}" for f in range(4)) + " |")
    print("|---|" + "---|" * 4)
    sel = dict(RC.selectors())["DB-only"]
    for label, pat in ARMS:
        cells = []
        for f in range(4):
            x = [r["correct"] for i, r in recs(pat, f).items() if sel(i)]
            cells.append(f"{100 * np.mean(x):.0f} ({len(x)})")
        print(f"| {label} | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    t1(); t2(); t3()
