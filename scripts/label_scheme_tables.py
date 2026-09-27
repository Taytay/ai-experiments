"""Tables for PLAN step 82 (MODEL-16): option labels for the one-slot reader (ai_experiments.oneslot): letters (A, B, C in order after the
shuffle, decider's), rand26 (a random sample of A..Z in random order per question), rand255 (random from decider's 255 single-token
labels, mostly two-letter). Qwen3.5-2B and decider-2B one slot + shot-label loss + database episodes, 800 steps, fold 0.

  L.1  per model, trained with and read with each scheme: REAL-6 scorecard, novel names top-1, and the flip rate against one reshuffle
       of the options (ORDER_SEED=1, the labels redrawn too)
usage: uv run python scripts/label_scheme_tables.py
"""
import sys
import warnings
from pathlib import Path

import numpy as np

from ai_experiments import real6 as R6
from ai_experiments import real6_cells as RC

sys.path.insert(0, str(Path(__file__).parent))
import open_models_tables as O  # noqa: E402

PRE = {"Qwen3.5-2B": "real6_dm_decider_slot_Qwen3.5-2B", "decider-2B": "real6_dm_decider_decider_decider-2b"}


def f(model, train, s, read):
    lab = "" if train == "letters" else f"_lab{train}"
    rd = "" if read == "letters" else f"_lab{read}"
    return f"{PRE[model]}_none_h100bf16st800_f0_dbep50_aux100{lab}_lora_{s}{rd}.noctx.jsonl"


def flips(a, b):
    ids = [i for i in a if i in b]
    return 100 * np.mean([int(np.argmax(a[i]["sum_lp"])) != int(np.argmax(b[i]["sum_lp"])) for i in ids]) if ids else float("nan")


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    r6 = {i["id"]: i for i in R6.load()["items"]}
    print("**Table L.1: one-slot readers by option-label scheme, REAL-6 fold 0 (trained with one scheme, read with one)**\n")
    print("| model | trained with | read with | n | top-1 [interval] | top-3 | bits left | auto-file at 98%: coverage (precision) | novel names top-1 | answers changed by a reshuffle |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for model in PRE:
        for train, read in (("letters", "letters"), ("letters", "rand26"), ("letters", "rand255"), ("rand26", "rand26"), ("rand255", "rand255")):
            recs = O.load(f(model, train, "real6", read))
            if not recs:
                continue
            nov = O.load(f(model, train, "real6_v1_novel", read))
            ordf = f(model, train, "real6_ord1", read)
            fl = flips({i: r for i, r in recs.items() if r6[i]["user"] % 4 == 0}, RC.load_recs(ordf)) if (RC.ROOT / "results" / "per_item" / ordf).exists() else float("nan")
            ex = f" {O.top1(nov, r6):.1f} | {fl:.1f}% |" if nov else f" – | {fl:.1f}% |"
            print(O.card_row(f"{model} | {train} | {read}", recs, r6, extra=ex))
