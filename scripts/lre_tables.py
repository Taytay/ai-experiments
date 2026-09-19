"""Table 40.1 for REPORT.md section 40 (PLAN step 28, GRAPH-1 / DATA-1) from results/lre_<tag>.json (scripts/exp_lre.py): per layer,
the linear relational probe's cross-validated accuracy on the trained species and its transfer to the held-out species, without and
with the field-guide entry in the prompt, for the type and weakness relations, and the weakness-via-type factoring, on each model.

usage: uv run python scripts/lre_tables.py [tag ...]      default tags: base, curriculum_Qwen2.5-3B_C_p200_lora, and their _wind counterparts
"""
import json
import sys

from ai_experiments.paths import ROOT

R = ROOT / "results"
DEFAULT = ["base", "curriculum_Qwen2.5-3B_C_p200_lora", "base_wind", "curriculum_Qwen2.5-3B_C_wind_p200_lora_wind"]
LABEL = {"base": "base", "curriculum_Qwen2.5-3B_C_p200_lora": "arm C", "base_wind": "base, independent weakness", "curriculum_Qwen2.5-3B_C_wind_p200_lora_wind": "arm C, independent weakness"}


def g(r, k):
    v = r.get(k)
    return "-" if v is None else f"{v:.0f}"


def main(tags):
    print("**Table 40.1: the linear relational probe (ridge map from the layer-l state at the answer position to the final state, decoded over the eight type "
          "options) per layer: 5-fold cross-validated accuracy on the 136 trained species / accuracy on the 24 held-out species, without and with the entry in "
          "context, for the type relation, the weakness relation, and weakness predicted by sending the type probe's answer through the rotation (accuracy %; chance 12.5; "
          "the 'model' row is the model's own answer at the final layer)**\n")
    for tag in tags:
        p = R / f"lre_{tag}.json"
        if not p.exists():
            print(f"\n{LABEL.get(tag, tag)}: no results"); continue
        d = json.load(open(p))
        layers = sorted({int(k.split("_")[0][1:]) for k in d["type_noctx"] if k.startswith("L") and k.endswith("_cv_acc")})
        print(f"\n*{LABEL.get(tag, tag)}* (model's own answer: type {d['type_noctx']['model_acc_seen']:.0f} / {d['type_noctx']['model_acc_held']:.0f} without context, "
              f"{d['type_ctx']['model_acc_seen']:.0f} / {d['type_ctx']['model_acc_held']:.0f} with; weakness {d['weakness_noctx']['model_acc_seen']:.0f} / {d['weakness_noctx']['model_acc_held']:.0f} and "
              f"{d['weakness_ctx']['model_acc_seen']:.0f} / {d['weakness_ctx']['model_acc_held']:.0f})\n")
        print("| layer | type: trained (cv) / held-out | type + context: trained / held-out | weakness: trained / held-out | weakness + context | weakness via type: trained / held-out | via type + context |")
        print("|---|---|---|---|---|---|---|")
        for l in layers:
            t, tc, w, wc = d["type_noctx"], d["type_ctx"], d["weakness_noctx"], d["weakness_ctx"]
            print(f"| {l} | {g(t, f'L{l}_cv_acc')} / {g(t, f'L{l}_held_acc')} | {g(tc, f'L{l}_cv_acc')} / {g(tc, f'L{l}_held_acc')} | {g(w, f'L{l}_cv_acc')} / {g(w, f'L{l}_held_acc')} | "
                  f"{g(wc, f'L{l}_cv_acc')} / {g(wc, f'L{l}_held_acc')} | {g(w, f'L{l}_via_type_acc')} / {g(w, f'L{l}_via_type_held_acc')} | {g(wc, f'L{l}_via_type_acc')} / {g(wc, f'L{l}_via_type_held_acc')} |")


if __name__ == "__main__":
    main(sys.argv[1:] or DEFAULT)
