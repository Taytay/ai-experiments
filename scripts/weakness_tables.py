"""Table 39.1 for REPORT.md section 39 (PLAN step 23, DATA-1): arm C on the original universe (weakness = a rotation of type, so the
weakness partition is the type partition) beside arm C on the universe where every species' weakness is drawn independently of its
type (names, habitats, diets and regions unchanged; items tagged _wind), with the untrained base on each. The bare recall level is
split by the asked attribute (type / weakness / habitat) from the per-item files, as in Table 30.1.

usage: uv run python scripts/weakness_tables.py            prints markdown; '-' where a run is missing
"""
import json

from ai_experiments.paths import ROOT

R = ROOT / "results"
ROWS = [("recall, trained fmt", "L1_recall_fmt"), ("recall, bare (pooled)", "L1_recall"), ("yes/no", "L2_manip_isa"), ("pair", "L2_manip_pair"),
        ("Timmy k=3 (type)", "L3_induct_type_nonsense"), ("k=4", "L3_induct_type_k4"), ("weakness induction", "L4_induct_weakness"), ("habitat induction", "L4_induct_habitat"),
        ("held-out species", "L3_induct_heldout"), ("v2 type (identifiable)", "I2_type"), ("v2 habitat", "I2_habitat"), ("ICL symbol", "ICL_symbol_mean"), ("ICL natural", "ICL_natural_mean"),
        ("ARC-Easy", "K_arc_easy"), ("WikiText ppl", "L7_ppl_wikitext"), ("training minutes", "train_minutes")]
COLS = [("base, original universe (sec. 8)", "curriculum_Qwen2.5-3B_base", "base", "ladder_v1.json"), ("C, original (sec. 15)", "curriculum_Qwen2.5-3B_C_p200", "trained", "ladder_v1.json"),
        ("C, original, fast path (sec. 19)", "curriculum_Qwen2.5-3B_C_fast_p200", "trained", "ladder_v1.json"),
        ("base, independent weakness", "curriculum_Qwen2.5-3B_base_wind", "base", "ladder_v1_wind.json"), ("C, independent weakness", "curriculum_Qwen2.5-3B_C_wind_p200", "trained", "ladder_v1_wind.json")]


def load(name):
    p = R / f"{name}.json"
    return json.load(open(p)) if p.exists() else {}


def fmt(v):
    return "-" if v is None else (f"{v:g}" if isinstance(v, float) else str(v))


def main():
    print("**Table 39.1: arm C with weakness as a rotation of type (every earlier section) and with weakness independent of type (PLAN step 23), same names and other attributes (accuracy %)**\n")
    print("| measure | " + " | ".join(c[0] for c in COLS) + " |")
    print("|---|" + "---|" * len(COLS))
    for label, key in ROWS:
        print(f"| {label} | " + " | ".join(fmt(load(f).get(c, {}).get(key)) for _, f, c, _l in COLS) + " |")
    for a in ("type", "weakness", "habitat"):
        cells = []
        for _, f, c, lad in COLS:
            pf = R / "per_item" / f"{f}.{c}.jsonl"; lp = ROOT / "data" / "processed" / lad
            if not pf.exists() or not lp.exists():
                cells.append("-"); continue
            attr = {it["id"]: it["attr"] for it in json.load(open(lp))["items"] if it["level"] == "L1_recall"}
            acc = [json.loads(l)["correct"] for l in open(pf) if (r := json.loads(l))["level"] == "L1_recall" and attr.get(r["id"]) == a]
            cells.append(f"{100 * sum(acc) / len(acc):.1f}" if acc else "-")
        print(f"| recall, bare: {a} questions | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()
