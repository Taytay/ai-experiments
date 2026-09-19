"""Table 42.1 for REPORT.md section 42 (PLAN step 30, GRAPH-4 / DATA-1) from results/graph4_<tag>.json: the induced type -> weakness edge,
bare (eight items, one per type; the count right is the number to read) and in the path form (136 items, the species' type given),
per adapter, on the original universe (where the edge exists) and the independent-weakness universe (where it does not).

usage: uv run python scripts/graph4_tables.py            prints markdown; '-' where a run is missing
"""
import json

from ai_experiments.paths import ROOT

R = ROOT / "results"
COLS = [("base", "base", "base"), ("A (knowledge only)", "curriculum_Qwen2.5-3B_A_p200_lora", "trained"), ("C", "curriculum_Qwen2.5-3B_C_p200_lora", "trained"),
        ("C, fast path", "curriculum_Qwen2.5-3B_C_fast_p200_lora", "trained"), ("D", "curriculum_Qwen2.5-3B_D_p200_lora", "trained"),
        ("C, 1,600 steps", "curriculum_Qwen2.5-3B_C_s1600_p400_lora", "trained"), ("C + OPD + replay", "curriculum_Qwen2.5-3B_C_opd_fineweb_full_replayC_lora", "trained"),
        ("base, independent weakness", "base_wind", "base"), ("C, independent weakness", "curriculum_Qwen2.5-3B_C_wind_p200_lora_wind", "trained")]


def load(tag):
    p = R / f"graph4_{tag}.json"
    return json.load(open(p)) if p.exists() else {}


def main():
    print("**Table 42.1: the induced type -> weakness edge. Bare: 'What type are T-type creatures weak to?' (eight items, count right of 8; chance 1). "
          "Path: the species' type stated, its weakness asked (136 items, accuracy %; chance 12.5; and the number of the eight types answered right on more than half of their species). "
          "On the independent-weakness universe the edge does not exist and the bare gold is the plurality weakness of the type's species (share 18 to 35%)**\n")
    print("| measure | " + " | ".join(c[0] for c in COLS) + " |")
    print("|---|" + "---|" * len(COLS))
    for label, key in (("bare: types right of 8", "G4_bare_correct_of_8"), ("path: accuracy", "G4_type_weakness_path"), ("path: types above half of 8", "G4_path_types_above_half")):
        cells = []
        for _, t, c in COLS:
            v = load(t).get(c, {}).get(key)
            cells.append("-" if v is None else (f"{v:g}" if isinstance(v, float) else str(v)))
        print(f"| {label} | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()
