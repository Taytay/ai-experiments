"""Tables for REPORT.md section 31 (PLAN step 31, TRAIN-8): on-policy distillation from the adapter-off base as the repair
for arm C's general-ability loss. Table 31.1 puts the repaired adapters (full ladder via `exp_curriculum.py C` with
EVAL_ONLY, results/curriculum_Qwen2.5-3B_C_opd_<tag>.json) beside the base and arm C (re-scored on 2026-09-18 to the same
numbers, so the originals stand) and arm Cg (general-text replay, TRAIN-7). Table 31.2 is the training
curve of each repair from results/opd_<tag>.json: reverse KL per step and the subsample points.

usage: uv run python scripts/opd_tables.py            prints markdown; '-' where a run is missing
"""
import json

from ai_experiments.paths import ROOT

R = ROOT / "results"

ROWS = [("recall, trained fmt", "L1_recall_fmt"), ("recall, bare", "L1_recall"), ("yes/no", "L2_manip_isa"), ("pair", "L2_manip_pair"),
        ("Timmy k=3", "L3_induct_type_nonsense"), ("k=4", "L3_induct_type_k4"), ("weakness", "L4_induct_weakness"), ("habitat", "L4_induct_habitat"),
        ("held-out species", "L3_induct_heldout"), ("unseen recall", "L6_unseen_recall"), ("seen recall control", "L6_seen_recall_ctrl"),
        ("reverse hard", "L8_reverse_hard"), ("ICL symbol", "ICL_symbol_mean"), ("ICL natural", "ICL_natural_mean"),
        ("ARC-Easy", "K_arc_easy"), ("WikiText ppl", "L7_ppl_wikitext")]
# the base and arm C were re-scored on 2026-09-18 (curriculum_Qwen2.5-3B_{base,C}_rescore0917.json) and reproduce the section 8 / 15
# numbers to the last decimal, so the original columns stand
COLS = [("base (sec. 8)", "curriculum_Qwen2.5-3B_base", "base"), ("C (sec. 15)", "curriculum_Qwen2.5-3B_C_p200", "trained"),
        ("Cg replay (sec. 17)", "curriculum_Qwen2.5-3B_Cg_p200", "trained"),
        ("C + OPD FineWeb, exact KL", "curriculum_Qwen2.5-3B_C_opd_fineweb_full", "trained"),
        ("C + OPD Tulu, exact KL", "curriculum_Qwen2.5-3B_C_opd_tulu_full", "trained"),
        ("C + OPD FineWeb + C replay", "curriculum_Qwen2.5-3B_C_opd_fineweb_full_replayC", "trained")]
CURVES = [("FineWeb, exact KL", "opd_fineweb_full"), ("Tulu, exact KL", "opd_tulu_full"), ("FineWeb + C replay", "opd_fineweb_full_replayC")]


def load(name):
    p = R / f"{name}.json"
    return json.load(open(p)) if p.exists() else {}


def fmt(v):
    return "-" if v is None else (f"{v:g}" if isinstance(v, float) else str(v))


def main():
    print("**Table 31.1: on-policy distillation (OPD) of arm C toward the adapter-off base on 120 steps of 64 prompts x 4 samples, full ladder "
          "(accuracy %; the base and arm C re-scored on 2026-09-18 reproduce their section 8 and 15 numbers exactly)**\n")
    print("| measure | " + " | ".join(c[0] for c in COLS) + " |")
    print("|---|" + "---|" * len(COLS))
    for label, key in ROWS:
        print(f"| {label} | " + " | ".join(fmt(load(f).get(c, {}).get(key)) for _, f, c in COLS) + " |")
    cells = []
    for name, f, _ in COLS:
        d = load(f.replace("curriculum_Qwen2.5-3B_C_", "")) if "opd" in f else {}  # results/opd_<tag>.json
        s = d.get("stats", {})
        cells.append(f"{s['train_minutes']} ({s['sampling_minutes']} sampling)" if s else "-")
    print("| repair minutes | " + " | ".join(cells) + " |")

    print("\n**Table 31.2: the repairs' curves: exact or sampled-token reverse KL per completion token (mean over the step's samples), mean sample length, "
          "and the subsample points (ladder[::4], ICL suite[::2], the 200 ARC-Easy items, the WikiText slice)**\n")
    print("| run | step | KL | replay loss | mean len | recall fmt | yes/no | Timmy k=3 | ICL sym | ICL nat | ARC-Easy | WikiText ppl |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for label, f in CURVES:
        d = load(f)
        if not d:
            print(f"| {label} | - | - | - | - | - | - | - | - | - | - |"); continue
        by_step = {c["step"]: c for c in d["curve"]}
        for s, m in sorted(((int(k), v) for k, v in d["periodic"].items())):
            c = by_step.get(s) or by_step.get(1) if s == 0 else by_step.get(s)
            kl = c.get("rkl_full", c["rkl_sampled"]) if c else None
            print(f"| {label} | {s} | {fmt(kl)} | {fmt(c.get('replay_loss')) if c else '-'} | {fmt(c['mean_len']) if c else '-'} | {fmt(m.get('L1_recall_fmt'))} | {fmt(m.get('L2_manip_isa'))} | "
                  f"{fmt(m.get('L3_induct_type_nonsense'))} | {fmt(m.get('ICL_symbol_mean'))} | {fmt(m.get('ICL_natural_mean'))} | {fmt(m.get('K_arc_easy'))} | {fmt(m.get('L7_ppl_wikitext'))} |")


if __name__ == "__main__":
    main()
