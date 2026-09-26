"""Table 34.1 for REPORT.md section 34 (PLAN step 21, DATA-3): arm E across marker shares (morph_p 0.3 to 1.0, suffix markers)
and on the prefix-marker universe, with the morphology probes of section 8 (M_probe_*: never-trained names built from trained
name parts) and the step 21 probes (M2_probe_*: an unseen prefix or suffix next to the marker), beside the untrained base on
each universe and the section 8 arm E adapter re-scored with the new probes.

usage: uv run python scripts/morph_tables.py            prints markdown; '-' where a run is missing
"""
import json

from ai_experiments.paths import ROOT

R = ROOT / "results"
ROWS = [("marked probe, trained fmt", "M_probe_marked_fmt"), ("plain probe, trained fmt", "M_probe_plain_fmt"), ("marked probe, bare", "M_probe_marked"), ("plain probe, bare", "M_probe_plain"),
        ("unseen-part marked probe, trained fmt", "M2_probe_marked_fmt"), ("unseen-part plain probe, trained fmt", "M2_probe_plain_fmt"),
        ("unseen-part marked probe, bare", "M2_probe_marked"), ("unseen-part plain probe, bare", "M2_probe_plain"),
        ("recall, trained fmt", "L1_recall_fmt"), ("recall, bare", "L1_recall"), ("yes/no", "L2_manip_isa"), ("Timmy k=3", "L3_induct_type_nonsense"), ("k=4", "L3_induct_type_k4"),
        ("held-out species", "L3_induct_heldout"), ("weakness (= type)", "L4_induct_weakness"), ("habitat", "L4_induct_habitat"),
        ("ICL symbol", "ICL_symbol_mean"), ("ICL natural", "ICL_natural_mean"), ("ARC-Easy", "K_arc_easy"), ("WikiText ppl", "L7_ppl_wikitext"), ("training minutes", "train_minutes")]
COLS = [("base, morph 0.7 (sec. 8)", "curriculum_Qwen2.5-3B_base_m", "base"), ("base, morph 0.7, re-scored", "curriculum_Qwen2.5-3B_base_m_probes2", "base"),
        ("E 0.7 (sec. 8)", "curriculum_Qwen2.5-3B_E", "trained"), ("E 0.7 (sec. 8), re-scored", "curriculum_Qwen2.5-3B_E_rescore", "trained"),
        ("E 0.3", "curriculum_Qwen2.5-3B_E_m30_p200", "trained"), ("E 0.5", "curriculum_Qwen2.5-3B_E_m50_p200", "trained"), ("E 0.7 fast path", "curriculum_Qwen2.5-3B_E_m70_p200", "trained"),
        ("E 0.9", "curriculum_Qwen2.5-3B_E_m90_p200", "trained"), ("E 1.0", "curriculum_Qwen2.5-3B_E_m100_p200", "trained"),
        ("base, prefix 0.7", "curriculum_Qwen2.5-3B_base_m_m70p", "base"), ("E prefix 0.7", "curriculum_Qwen2.5-3B_E_m70p_p200", "trained"),
        ("base, plain", "curriculum_Qwen2.5-3B_base_probes2", "base"), ("C, plain (sec. 15)", "curriculum_Qwen2.5-3B_C_p200", "trained")]


def load(name):
    p = R / f"{name}.json"
    return json.load(open(p)) if p.exists() else {}


def fmt(v):
    return "-" if v is None else (f"{v:g}" if isinstance(v, float) else str(v))


def main():
    print("**Table 34.1: arm E (the section 8 mixture on a morphology universe) across marker shares and marker positions, with the section 8 probes "
          "(never-trained names from trained name parts) and the step 21 probes (an unseen prefix or suffix beside the marker); chance 12.5 on every probe level (accuracy %)**\n")
    print("| measure | " + " | ".join(c[0] for c in COLS) + " |")
    print("|---|" + "---|" * len(COLS))
    for label, key in ROWS:
        print(f"| {label} | " + " | ".join(fmt(load(f).get(c, {}).get(key)) for _, f, c in COLS) + " |")


if __name__ == "__main__":
    main()
