"""Emit the section 8 markdown tables from results/curriculum_Qwen2.5-3B_<arm>.json.

usage: uv run python scripts/gen_section8_tables.py"""
import json
from pathlib import Path

R = Path(__file__).parent.parent / "results"
tag = "Qwen2.5-3B"


def load(arm):
    d = json.loads((R / f"curriculum_{tag}_{arm}.json").read_text())
    return d.get("trained", d.get("base")), d.get("trained_ctx", d.get("base_ctx"))


ARMS = [("base", "base"), ("A", "A knowledge"), ("B", "B episodes"), ("C", "C interleaved + replay"),
        ("Cn", "Cn interleaved"), ("D", "D sequential")]
data = {a: load(a) for a, _ in ARMS}


def table(title, cols, cond, arms=ARMS):
    print(f"**{title}**\n")
    print("| arm | " + " | ".join(c[1] for c in cols) + " |")
    print("|---|" + "---|" * len(cols))
    for a, lab in arms:
        row = data[a][cond]
        print(f"| {lab} | " + " | ".join(f"{row.get(c[0], ''):g}" if isinstance(row.get(c[0]), (int, float)) else "" for c in cols) + " |")
    print()


table("Table 8.1: from the weights, no context (accuracy %)", [
    ("L1_recall_fmt", "recall (trained fmt)"), ("L2_manip_isa", "yes/no"), ("L2_manip_pair", "pair"),
    ("L3_induct_type_nonsense", "Timmy k=3"), ("L3_induct_type_k2", "k=2"), ("L3_induct_type_k4", "k=4"),
    ("L3_induct_type_realnames", "real-name labels"), ("L4_induct_weakness", "weakness (held-out attr)"),
    ("L4_induct_habitat", "habitat"), ("L3_induct_heldout", "held-out species"),
    ("ICL_symbol_mean", "ICL suite symbol"), ("ICL_natural_mean", "ICL suite natural"), ("L7_ppl_general", "ppl")], 0)

table("Table 8.2: with field-guide context (accuracy %)", [
    ("L1_recall_fmt", "recall (trained fmt)"), ("L2_manip_isa", "yes/no"), ("L2_manip_pair", "pair"),
    ("L3_induct_type_nonsense", "Timmy k=3"), ("L3_induct_type_k4", "k=4"), ("L3_induct_type_realnames", "real-name labels"),
    ("L4_induct_weakness", "weakness (held-out attr)"), ("L4_induct_habitat", "habitat"), ("L3_induct_heldout", "held-out species")], 1)

MARMS = [("base", "base, plain universe"), ("C", "C, plain universe"), ("base_m", "base, morphology universe"), ("E", "E interleaved, morphology universe")]
data.update({a: load(a) for a, _ in MARMS if a not in data})
table("Table 8.3: morphology universe (70% of names carry a type suffix) vs plain, no context (accuracy %)", [
    ("L1_recall_fmt", "recall (trained fmt)"), ("L3_induct_type_nonsense", "Timmy k=3"), ("L3_induct_type_k4", "k=4"),
    ("L4_induct_weakness", "weakness"), ("L3_induct_heldout", "held-out species"),
    ("M_probe_marked_fmt", "probe: marked name"), ("M_probe_plain_fmt", "probe: plain name"),
    ("M_probe_marked", "marked (bare fmt)"), ("M_probe_plain", "plain (bare fmt)"),
    ("ICL_symbol_mean", "ICL suite symbol"), ("L7_ppl_general", "ppl")], 0, MARMS)
