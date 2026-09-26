"""Tables for REPORT.md section 32 (PLAN step 32, TRAIN-9): LoRA at ten times the full fine-tuning rate. Table 32.1 is arm C
(Qwen2.5-3B, rank 64, 800 steps, fast path, seed 0) across the learning rates of section 28 (5e-5, 1e-4, 2e-4) and this step
(5e-4, 1e-3), with the 1,600-step run at 1e-4 beside them; Table 32.2 is the Flan-T5 F2A adapter at 3e-4 (section 29), 1e-3 and
3e-3 beside the full fine-tunes at 1e-4 / 3e-4 / 1e-3 and the T5Gemma adapters at 1e-4 / 3e-4.

usage: uv run python scripts/lr_tables.py            prints markdown; '-' where a run is missing
"""
import json

from ai_experiments.paths import ROOT

R = ROOT / "results"

ROWS = [("recall, trained fmt", "L1_recall_fmt"), ("recall, bare", "L1_recall"), ("yes/no", "L2_manip_isa"), ("pair", "L2_manip_pair"),
        ("Timmy k=3", "L3_induct_type_nonsense"), ("k=4", "L3_induct_type_k4"), ("weakness", "L4_induct_weakness"), ("habitat", "L4_induct_habitat"),
        ("held-out species", "L3_induct_heldout"), ("seen recall control", "L6_seen_recall_ctrl"), ("reverse hard", "L8_reverse_hard"),
        ("ICL symbol", "ICL_symbol_mean"), ("ICL natural", "ICL_natural_mean"), ("ARC-Easy", "K_arc_easy"), ("WikiText ppl", "L7_ppl_wikitext"),
        ("training minutes", "train_minutes")]
COLS_C = [("base", "curriculum_Qwen2.5-3B_base", "base"), ("5e-5", "curriculum_Qwen2.5-3B_C_lr5e-5_p200", "trained"),
          ("1e-4 (recipe)", "curriculum_Qwen2.5-3B_C_fast_p200", "trained"), ("2e-4", "curriculum_Qwen2.5-3B_C_lr2e-4_p200", "trained"),
          ("5e-4", "curriculum_Qwen2.5-3B_C_lr5e-4_p200", "trained"), ("1e-3", "curriculum_Qwen2.5-3B_C_lr1e-3_p200", "trained"),
          ("1e-4, 1,600 steps", "curriculum_Qwen2.5-3B_C_s1600_p400", "trained")]

ROWS_E = [("recall, trained fmt", "L1_recall_fmt", True), ("recall, bare", "L1_recall", True), ("yes/no", "L2_manip_isa", True), ("pair", "L2_manip_pair", True),
          ("Timmy k=3", "L3_induct_type_nonsense", False), ("reverse hard", "L8_reverse_hard", True), ("ICL symbol", "ICL_symbol_mean", False),
          ("ICL natural", "ICL_natural_mean", False), ("ARC-Easy", "K_arc_easy", False), ("training minutes", "train_minutes", False)]
COLS_E = [("Flan-T5 base", "flan-t5-large_base", "base"), ("full FT 1e-4", "flan-t5-large_F2A", "trained"), ("full FT 3e-4", "flan-t5-large_F2A_lr3e-4", "trained"),
          ("full FT 1e-3", "flan-t5-large_F2A_lr1e-3", "trained"), ("LoRA 3e-4", "flan-t5-large_F2A_lora3e-4", "trained"), ("LoRA 1e-3", "flan-t5-large_F2A_lora1e-3", "trained"),
          ("LoRA 3e-3", "flan-t5-large_F2A_lora3e-3", "trained"), ("T5Gemma LoRA 1e-4 (F2)", "t5gemma-l-l-ul2_F2_lora", "trained"), ("T5Gemma LoRA 3e-4 (F2)", "t5gemma-l-l-ul2_F2_lora3e-4", "trained")]


def load(name):
    p = R / f"{name}.json"
    return json.load(open(p)) if p.exists() else {}


def fmt(v):
    return "-" if v is None else (f"{v:g}" if isinstance(v, float) else str(v))


def main():
    print("**Table 32.1: arm C on Qwen2.5-3B (rank 64, 800 steps unless stated, seed 0) across learning rates; the recipe's 1e-4 is the full-fine-tuning rate, "
          "Tinker's LoRA rate is ten times that (accuracy %)**\n")
    print("| measure | " + " | ".join(c[0] for c in COLS_C) + " |")
    print("|---|" + "---|" * len(COLS_C))
    for label, key in ROWS:
        print(f"| {label} | " + " | ".join(fmt(load(f).get(c, {}).get(key)) for _, f, c in COLS_C) + " |")

    print("\n**Table 32.2: the Flan-T5 knowledge-only arm (F2A, span prediction) under full fine-tuning and the rank-64 adapter across learning rates, "
          "with the T5Gemma adapters for comparison (span-format value in brackets)**\n")
    print("| measure | " + " | ".join(c[0] for c in COLS_E) + " |")
    print("|---|" + "---|" * len(COLS_E))
    for label, key, span in ROWS_E:
        cells = []
        for _, f, c in COLS_E:
            d = load(f"encoder_{f}")
            v, s = fmt(d.get(c, {}).get(key)), fmt(d.get(c + "_span", {}).get(key))
            cells.append(v + (f" [{s}]" if span and s != "-" else ""))
        print(f"| {label} | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()
