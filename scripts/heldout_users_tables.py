"""Tables for REPORT.md section 51 (PLAN step 42, REAL-10): the categoriser on users whose schemes it never trained on. Each fold adapter
(users with id mod 4 = k held out) is scored on its five held-out users; the four folds merge into one held-out reading of all 1,179
items, paired item by item with the adapter trained on all 20 users.

  51.1  all 20 users trained against user held out: the no-DB arm (800 steps, REPORT.md 49) and the record arm (200 steps), by name type
        and REPORT.md 48's groups, with the user-resampled interval
  51.2  paired held out minus all-20, by name type
  51.3  rename augmentation (RENAME=0.5: each category name replaced by a fresh coined word with probability 0.5 per episode) on held-out
        users: the all-label no-DB recipe of section 52 and the record arm; paired with the same recipe without it, by name type
All runs on the 3090 (4-bit base, 4 x 4 sequences per step).
usage: uv run python scripts/heldout_users_tables.py
"""
from ai_experiments import real6_cells as RC

CAT = "real6_categoriser_Qwen2.5-3B-Instruct"
COLS = ["all", "standard", "renamed", "coined", "in history", "determined by category", "split category", "DB-only"]
ARMS = [("SFT no DB, 800 steps, all 20 users trained", f"{CAT}_none_st800_s0_lora.noctx.jsonl"),
        ("SFT no DB, 800 steps, user held out", f"{CAT}_none_st800_f?_lora.noctx.jsonl"),
        ("SFT + record, 200 steps, all 20 users trained", f"{CAT}_ret_lora.ctx.jsonl"),
        ("SFT + record, 200 steps, user held out", f"{CAT}_ret_f?_lora.ctx.jsonl")]
RENAME = [("no DB, all-label 200 steps, held out", f"{CAT}_none_f?_alllab_lora.noctx.jsonl", f"{CAT}_none_f?_ren50_alllab_lora.noctx.jsonl"),
          ("SFT + record, 200 steps, held out", f"{CAT}_ret_f?_lora.ctx.jsonl", f"{CAT}_ret_f?_ren50_lora.ctx.jsonl")]


def t1():
    print("**Table 51.1: trained on all 20 users against the user held out (accuracy %; held-out rows merge the four fold adapters, each scoring "
          "the five users it never trained on; interval = users resampled)**\n")
    print("| arm | " + " | ".join(COLS) + " | interval (all) |")
    print("|---|" + "---|" * (len(COLS) + 1))
    for label, pat in ARMS:
        r = RC.load_recs(pat); lo, hi = RC.user_ci({i: x["correct"] for i, x in r.items()})
        print(f"| {label} | " + " | ".join(RC.row_cells(r, COLS)) + f" | [{lo}, {hi}] |")


def t2():
    print("\n**Table 51.2: held out minus all-20 on the same items, points [user-resampled 95% interval]**\n")
    cols = ["all", "standard", "renamed", "coined", "in history"]
    print("| arm | " + " | ".join(cols) + " |")
    print("|---|" + "---|" * len(cols))
    for (label, a), (_, b) in ((ARMS[0], ARMS[1]), (ARMS[2], ARMS[3])):
        A, B = RC.load_recs(a), RC.load_recs(b)
        print(f"| {label.split(',')[0]} | " + " | ".join(RC.paired(A, B, c) for c in cols) + " |")


def t3():
    print("\n**Table 51.3: rename augmentation on held-out users (accuracy %, then with minus without on the same items [interval])**\n")
    cols = ["all", "standard", "renamed", "coined"]
    print("| recipe | augmentation | " + " | ".join(cols) + " |")
    print("|---|---|" + "---|" * len(cols))
    for label, plain, ren in RENAME:
        P, R = RC.load_recs(plain), RC.load_recs(ren)
        print(f"| {label} | without | " + " | ".join(RC.row_cells(P, cols)) + " |")
        print(f"| {label} | with | " + " | ".join(RC.row_cells(R, cols)) + " |")
        print(f"| {label} | with minus without | " + " | ".join(RC.paired(P, R, c) for c in cols) + " |")


if __name__ == "__main__":
    t1(); t2(); t3()
