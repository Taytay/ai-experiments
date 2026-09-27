"""Tables for PLAN step 67 (MODEL-8): per-layer linear probes on Qwen3.5-2B (results/probes_*.json from scripts/exp_probes.py).
  P.1  the merchant's standard category at the statement position: cross-validated by merchant on labelled merchants / read on the
       DB-only merchants (a probe trained on the others), per model and layer
  P.2  the same at the "Category:" cue, with the gold label's standard category (cross-validated by user)
usage: uv run python scripts/probes_tables.py
"""
import json

from ai_experiments.paths import ROOT

MODELS = [("untrained", "probes_Qwen3.5-2B.json"), ("trained, no database", "probes_categoriser_Qwen3.5-2B_none_h100bf16_hf_f0_alllab_lora.json"),
          ("trained, database episodes", "probes_categoriser_Qwen3.5-2B_none_h100bf16_hf_f0_alllab_dbep50_lora.json")]

if __name__ == "__main__":
    D = [(lab, json.loads((ROOT / "results" / f).read_text())) for lab, f in MODELS]
    L = D[0][1]["layers"]; layers = list(range(0, L, 3)) + ([L - 1] if (L - 1) % 3 else [])
    for k, (pos, title) in enumerate((("stmt", "the last token of the query's statement string"), ("cue", 'the final "Category:" token'))):
        print(f"**Table P.{k + 1}: probe accuracy (%) at {title}, by layer (0 = embeddings): the merchant's standard category, "
              f"cross-validated by merchant / read on the {D[0][1]['n_db_only']} DB-only items" + (" / the gold label's standard category" if pos == "cue" else "") + "**\n")
        print("| layer | " + " | ".join(lab for lab, _ in D) + " |"); print("|---|" + "---|" * len(D))
        for l in layers:
            cells = []
            for _, d in D:
                r = d["probes"][f"{pos}_{l}"]
                cells.append(f"{100 * r['std_cv']:.0f} / {100 * r['std_db_only']:.0f}" + (f" / {100 * r['gold_cv']:.0f}" if "gold_cv" in r else ""))
            print(f"| {l} | " + " | ".join(cells) + " |")
        print()
