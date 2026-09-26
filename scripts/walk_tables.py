"""Table 41.1 for REPORT.md section 41 (PLAN step 29, GRAPH-3): arm Cw (a ninth of the knowledge sequences replaced by two-hop walk texts,
universe.walk_texts) beside arm C on the same fast path and the section 15 run, on the ladder levels the walks are meant to move (pair,
yes/no, Timmy) and the rest, with the per-stream token counts the tracker recorded.

usage: uv run python scripts/walk_tables.py            prints markdown; '-' where a run is missing
"""
import json

from ai_experiments.paths import ROOT

R = ROOT / "results"
ROWS = [("recall, trained fmt", "L1_recall_fmt"), ("recall, bare", "L1_recall"), ("yes/no", "L2_manip_isa"), ("pair", "L2_manip_pair"),
        ("Timmy k=3", "L3_induct_type_nonsense"), ("k=4", "L3_induct_type_k4"), ("habitat induction", "L4_induct_habitat"), ("held-out species", "L3_induct_heldout"),
        ("v2 type (identifiable)", "I2_type"), ("v2 habitat", "I2_habitat"), ("v2 diet", "I2_diet"), ("v2 region", "I2_region"), ("unseen label", "I2_type_unseen"),
        ("reverse hard", "L8_reverse_hard"), ("ICL symbol", "ICL_symbol_mean"), ("ICL natural", "ICL_natural_mean"), ("ARC-Easy", "K_arc_easy"), ("MMLU", "K_mmlu"),
        ("WikiText ppl", "L7_ppl_wikitext"), ("knowledge tokens (K)", "tok_K"), ("walk tokens (W)", "tok_W"), ("episode tokens (E)", "tok_E"), ("training minutes", "train_minutes")]
COLS = [("base", "curriculum_Qwen2.5-3B_base", "base"), ("C (sec. 15)", "curriculum_Qwen2.5-3B_C_p200", "trained"), ("C, fast path (sec. 19)", "curriculum_Qwen2.5-3B_C_fast_p200", "trained"),
        ("C, 1,600 steps (sec. 28)", "curriculum_Qwen2.5-3B_C_s1600_p400", "trained"), ("Cw: walks in the knowledge stream", "curriculum_Qwen2.5-3B_Cw_p200", "trained")]


def load(name):
    p = R / f"{name}.json"
    return json.load(open(p)) if p.exists() else {}


def fmt(v):
    return "-" if v is None else (f"{v:,}" if isinstance(v, int) else (f"{v:g}" if isinstance(v, float) else str(v)))


def main():
    print("**Table 41.1: arm Cw (knowledge .40, two-hop walk texts .05, episodes .40, replay .15) against arm C (knowledge .45), 800 steps, seed 0 (accuracy %)**\n")
    print("| measure | " + " | ".join(c[0] for c in COLS) + " |")
    print("|---|" + "---|" * len(COLS))
    for label, key in ROWS:
        print(f"| {label} | " + " | ".join(fmt(load(f).get(c, {}).get(key)) for _, f, c in COLS) + " |")


if __name__ == "__main__":
    main()
