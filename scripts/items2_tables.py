"""Table 33.1 for REPORT.md section 33 (PLAN step 20, EVAL-4 / EVAL-5): the v2 item sets scored on saved weights by
`scripts/exp_items_v2.py` (results/items2_<tag>.json): identifiable induction items without and with the field guide in
context, the unseen-label variant, the out-of-distribution ICL suite, the MMLU slice, beside the v1 induction levels, the v1
suite and ARC-Easy of the same weights. Table 33.2 is `scripts/induct_strata.py`.

usage: uv run python scripts/items2_tables.py            prints markdown; '-' where a run is missing
"""
import json

from ai_experiments.paths import ROOT

R = ROOT / "results"
COLS = [("base", "base", "curriculum_Qwen2.5-3B_base", "base"), ("A", "curriculum_Qwen2.5-3B_A_p200_lora", "curriculum_Qwen2.5-3B_A_p200", "trained"),
        ("B", "curriculum_Qwen2.5-3B_B_lora", "curriculum_Qwen2.5-3B_B", "trained"), ("C", "curriculum_Qwen2.5-3B_C_p200_lora", "curriculum_Qwen2.5-3B_C_p200", "trained"),
        ("D", "curriculum_Qwen2.5-3B_D_p200_lora", "curriculum_Qwen2.5-3B_D_p200", "trained"), ("Cg", "curriculum_Qwen2.5-3B_Cg_p200_lora", "curriculum_Qwen2.5-3B_Cg_p200", "trained"),
        ("C 2e-4", "curriculum_Qwen2.5-3B_C_lr2e-4_p200_lora", "curriculum_Qwen2.5-3B_C_lr2e-4_p200", "trained"),
        ("C 1,600 steps", "curriculum_Qwen2.5-3B_C_s1600_p400_lora", "curriculum_Qwen2.5-3B_C_s1600_p400", "trained"),
        ("C + OPD", "curriculum_Qwen2.5-3B_C_opd_fineweb_full_lora", "curriculum_Qwen2.5-3B_C_opd_fineweb_full", "trained"),
        ("C + OPD + replay", "curriculum_Qwen2.5-3B_C_opd_fineweb_full_replayC_lora", "curriculum_Qwen2.5-3B_C_opd_fineweb_full_replayC", "trained"),
        ("AlphaEdit type", "edit_alphaedit_type_qwen2.5-3b", "curriculum_edit_alphaedit_type_qwen2.5-3b_base", "base")]
# (label, key, source): source "v2" = items2 file, "v2ctx" = its ctx condition, "v1" = the curriculum file of the same weights
ROWS = [("v1 Timmy k=3 (type, 1 demo/group)", "L3_induct_type_nonsense", "v1"), ("v1 habitat", "L4_induct_habitat", "v1"), ("v1 weakness (= type)", "L4_induct_weakness", "v1"),
        ("v2 type (2 demos/group, identifiable)", "I2_type", "v2"), ("v2 habitat", "I2_habitat", "v2"), ("v2 diet", "I2_diet", "v2"), ("v2 region", "I2_region", "v2"),
        ("v2 type, unseen label", "I2_type_unseen", "v2"),
        ("v2 type + context", "I2_type", "v2ctx"), ("v2 habitat + context", "I2_habitat", "v2ctx"), ("v2 diet + context", "I2_diet", "v2ctx"), ("v2 region + context", "I2_region", "v2ctx"),
        ("v2 type, unseen label + context", "I2_type_unseen", "v2ctx"),
        ("v1 ICL symbol", "ICL_symbol_mean", "v2"), ("v2 ICL symbol (Q/A, rare words)", "ICL2_symbol_mean", "v2"),
        ("v1 ICL natural", "ICL_natural_mean", "v2"), ("v2 ICL natural (Q/A)", "ICL2_natural_mean", "v2"),
        ("ARC-Easy", "K_arc_easy", "v2"), ("MMLU 5-shot", "K_mmlu", "v2"), ("WikiText ppl", "L7_ppl_wikitext", "v1")]


def load(name):
    p = R / f"{name}.json"
    return json.load(open(p)) if p.exists() else {}


def fmt(v):
    return "-" if v is None else (f"{v:g}" if isinstance(v, float) else str(v))


def main():
    print("**Table 33.1: the step 20 item sets on the saved weights (accuracy %; chance 33.3 on the k=3 induction levels, 25 on ARC-Easy and MMLU, "
          "the ICL suites' chance is the mean of 1/k). v1 rows are the same weights' original ladder scores**\n")
    print("| measure | " + " | ".join(c[0] for c in COLS) + " |")
    print("|---|" + "---|" * len(COLS))
    for label, key, src in ROWS:
        cells = []
        for _, tag, v1file, v1cond in COLS:
            if src == "v1":
                v = load(v1file).get(v1cond, {}).get(key)
            else:
                d = load(f"items2_{tag}")
                cond = ("base" if tag == "base" else "trained") + ("_ctx" if src == "v2ctx" else "")  # exp_items_v2 files a full checkpoint under "trained"
                v = d.get(cond, {}).get(key)
            cells.append(fmt(v))
        print(f"| {label} | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()
