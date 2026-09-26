"""Tables 29.2b and 29.3 of reports/REPORT.md from results/encoder_*.json (PLAN step 18, MODEL-2): the encoder-decoders'
fact levels in their trained span-prediction format, and Flan-T5 / T5Gemma across universe sizes, learning rates and the
rank-64 adapter. Prints markdown; '-' where a run or a scoring condition is missing.

usage: uv run python scripts/encoder_tables.py
"""
import json

from ai_experiments.paths import ROOT

R = ROOT / "results"


def load(name):
    p = R / f"encoder_{name}.json"
    return json.load(open(p)) if p.exists() else {}


def g(d, cond, key):
    v = d.get(cond, {}).get(key)
    return "-" if v is None else (f"{v:g}" if isinstance(v, float) else str(v))


ROWS_SPAN = [("recall, trained fmt", "L1_recall_fmt"), ("recall, bare", "L1_recall"), ("yes/no", "L2_manip_isa"), ("pair", "L2_manip_pair"),
             ("reverse easy", "L8_reverse_easy"), ("reverse hard", "L8_reverse_hard"), ("generation: recall fmt", "gen_L1_recall_fmt"),
             ("generation: recall bare", "gen_L1_recall"), ("generation: reverse hard", "gen_L8_reverse_hard")]
COLS_SPAN = [("Flan-T5 base", "flan-t5-large_base", "base"), ("Flan-T5 F2A (knowledge)", "flan-t5-large_F2A", "trained"),
             ("Flan-T5 F2 (mixture)", "flan-t5-large_F2", "trained"), ("T5Gemma base", "t5gemma-l-l-ul2_base", "base"),
             ("T5Gemma F2 full FT 1e-4", "t5gemma-l-l-ul2_F2", "trained"), ("T5Gemma F2 LoRA 1e-4", "t5gemma-l-l-ul2_F2_lora", "trained"),
             ("T5Gemma F2 LoRA 3e-4", "t5gemma-l-l-ul2_F2_lora3e-4", "trained"), ("Flan-T5 F2A LoRA 3e-4", "flan-t5-large_F2A_lora3e-4", "trained")]

ROWS3 = [("recall, trained fmt", "L1_recall_fmt", True), ("recall, bare", "L1_recall", True), ("generation: recall fmt", "gen_L1_recall_fmt", True),
         ("yes/no", "L2_manip_isa", True), ("Timmy k=3", "L3_induct_type_nonsense", False), ("reverse hard", "L8_reverse_hard", True),
         ("ICL symbol", "ICL_symbol_mean", False), ("ICL natural", "ICL_natural_mean", False), ("ARC-Easy", "K_arc_easy", False),
         ("training minutes", "train_minutes", False)]
COLS3 = [("160 / 800", "flan-t5-large_F2"), ("1,000 / 800", "flan-t5-large_F2_n1000"), ("1,000 / 4,000", "flan-t5-large_F2_n1000x5"),
         ("5,000 / 4,000", "flan-t5-large_F2_n5000x5"), ("F2A 160 lr 3e-4", "flan-t5-large_F2A_lr3e-4"), ("F2A 160 lr 1e-3", "flan-t5-large_F2A_lr1e-3"),
         ("F2A 160 lr 1e-3 / 4,000", "flan-t5-large_F2A_lr1e-3x5"), ("F2A 160 LoRA 3e-4", "flan-t5-large_F2A_lora3e-4"),
         ("T5Gemma F2 LoRA 1e-4", "t5gemma-l-l-ul2_F2_lora"), ("T5Gemma F2 LoRA 3e-4", "t5gemma-l-l-ul2_F2_lora3e-4")]


def main():
    print("**Table 29.2b: the encoder-decoders' fact levels in the span-prediction format they were trained in (option log-probability; generation in the gen rows)**\n")
    print("| measure | " + " | ".join(c[0] for c in COLS_SPAN) + " |")
    print("|---|" + "---|" * len(COLS_SPAN))
    for label, key in ROWS_SPAN:
        print(f"| {label} | " + " | ".join(g(load(f), c + "_span", key) for _, f, c in COLS_SPAN) + " |")

    print("\n**Table 29.3: Flan-T5 at 1,000 and 5,000 species, the knowledge-only arm at T5's usual learning rates, and the rank-64 adapter runs on both encoder-decoders (span-format value in brackets)**\n")
    print("| measure | " + " | ".join(c[0] for c in COLS3) + " |")
    print("|---|" + "---|" * len(COLS3))
    for label, key, span in ROWS3:
        cells = []
        for _, f in COLS3:
            d = load(f)
            v, s = g(d, "trained", key), g(d, "trained_span", key)
            cells.append(v + (f" [{s}]" if span and s != "-" else ""))
        print(f"| {label} | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()
