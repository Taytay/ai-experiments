"""Table for REPORT.md section 30 (PLAN step 19, BASE-2): the knowledge-editing arms (MEMIT / AlphaEdit through EasyEdit on
Qwen2.5-3B, one edit per trained species for the type, or five for type / weakness / habitat / diet / region) scored on the
full ladder and suite by `scripts/exp_curriculum.py base <edited model dir>`, beside the untrained base and arms A and C.

usage: uv run python scripts/edit_tables.py            prints markdown; '-' where a run is missing
"""
import json

from ai_experiments.paths import ROOT

R = ROOT / "results"

ROWS = [("recall, trained fmt", "L1_recall_fmt"), ("recall, bare", "L1_recall"), ("yes/no", "L2_manip_isa"), ("pair", "L2_manip_pair"),
        ("Timmy k=3", "L3_induct_type_nonsense"), ("k=4", "L3_induct_type_k4"), ("weakness", "L4_induct_weakness"), ("habitat", "L4_induct_habitat"),
        ("held-out species", "L3_induct_heldout"), ("unseen recall", "L6_unseen_recall"), ("seen recall control", "L6_seen_recall_ctrl"),
        ("reverse easy", "L8_reverse_easy"), ("reverse hard", "L8_reverse_hard"), ("ICL symbol", "ICL_symbol_mean"), ("ICL natural", "ICL_natural_mean"),
        ("ARC-Easy", "K_arc_easy"), ("WikiText ppl", "L7_ppl_wikitext")]
COLS = [("base", "curriculum_Qwen2.5-3B_base", "base"), ("A (knowledge)", "curriculum_Qwen2.5-3B_A_p200", "trained"), ("C (mixture)", "curriculum_Qwen2.5-3B_C_p200", "trained"),
        ("MEMIT type", "curriculum_edit_memit_type_qwen2.5-3b_base", "base"), ("MEMIT type, ridge 150", "curriculum_edit_memit_type_ridge150_qwen2.5-3b_base", "base"), ("MEMIT type, ridge 15", "curriculum_edit_memit_type_ridge15_qwen2.5-3b_base", "base"), ("AlphaEdit type", "curriculum_edit_alphaedit_type_qwen2.5-3b_base", "base"),
        ("MEMIT all relation-first, ridge 15", "curriculum_edit_memit_allpre_ridge15_qwen2.5-3b_base", "base"), ("AlphaEdit all", "curriculum_edit_alphaedit_all_qwen2.5-3b_base", "base"),
        ("MEMIT all record relation-first, ridge 15", "curriculum_edit_memit_allyaml_ridge15_qwen2.5-3b_base", "base"), ("MEMIT all record last-token key, ridge 15", "curriculum_edit_memit_allyamllast_ridge15_qwen2.5-3b_base", "base")]
EDITS = {"MEMIT type": "edit_memit_type", "MEMIT type, ridge 150": "edit_memit_type_ridge150", "MEMIT type, ridge 15": "edit_memit_type_ridge15", "AlphaEdit type": "edit_alphaedit_type", "MEMIT all relation-first, ridge 15": "edit_memit_allpre_ridge15", "AlphaEdit all": "edit_alphaedit_all", "MEMIT all record relation-first, ridge 15": "edit_memit_allyaml_ridge15", "MEMIT all record last-token key, ridge 15": "edit_memit_allyamllast_ridge15"}


def load(name):
    p = R / f"{name}.json"
    return json.load(open(p)) if p.exists() else {}


def fmt(v):
    return "-" if v is None else (f"{v:g}" if isinstance(v, float) else str(v))


def main():
    print("**Table 30.1: knowledge editing on Qwen2.5-3B (EasyEdit MEMIT, MEMIT with a ridge on the solve, and AlphaEdit; layers 4-8, one batch) against the untrained base and the fine-tuned arms A and C on the 160-species items (accuracy %)**\n")
    print("| measure | " + " | ".join(c[0] for c in COLS) + " |")
    print("|---|" + "---|" * len(COLS))
    for label, key in ROWS:
        print(f"| {label} | " + " | ".join(fmt(load(f).get(c, {}).get(key)) for _, f, c in COLS) + " |")
    # bare recall split by the asked attribute (the ladder mixes type, weakness and habitat questions; the type-only edits answer
    # the type question and mis-fire on the weakness one, which the pooled number hides)
    import collections
    lad = json.load(open(ROOT / "data" / "processed" / "ladder_v1.json"))["items"]
    attr = {it["id"]: it["attr"] for it in lad if it["level"] == "L1_recall"}
    for a in ("type", "weakness", "habitat"):
        cells = []
        for _, f, c in COLS:
            pf = R / "per_item" / f"{f}.{c}.jsonl"
            if not pf.exists():
                cells.append("-"); continue
            acc = [json.loads(l)["correct"] for l in open(pf) if (r := json.loads(l))["level"] == "L1_recall" and attr.get(r["id"]) == a]
            cells.append(f"{100 * sum(acc) / len(acc):.1f}" if acc else "-")
        print(f"| recall, bare: {a} questions | " + " | ".join(cells) + " |")
    for label, key in (("edits", "n_edits"), ("edit minutes", "edit_minutes"), ("EasyEdit rewrite acc", "rewrite_acc"), ("EasyEdit rephrase acc", "rephrase_acc")):
        cells = []
        for name, _, _ in COLS:
            e = load(EDITS[name]) if name in EDITS else {}
            v = e.get(key, e.get("easyedit_post", {}).get(key)) if e else None
            cells.append(fmt(v))
        print(f"| {label} | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()
