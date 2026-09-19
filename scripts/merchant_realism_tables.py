"""Tables for REPORT.md section 35 (PLAN step 36, DATA-4 / DATA-2) from results/merchant_realism.json: the embedding route and the
0.5B route on the v1 and v2 merchant sets, trained on the section 4 text alone (clean) or with six card-statement renderings per
merchant (rend), scored on the held-out bank string, its normalised form, the truncated string and its normalised form.

usage: uv run python scripts/merchant_realism_tables.py            prints markdown; '-' where a run is missing
"""
import json

from ai_experiments.paths import ROOT

R = json.load(open(ROOT / "results" / "merchant_realism.json")) if (ROOT / "results" / "merchant_realism.json").exists() else {}


def g(key, k):
    v = R.get(key, {}).get(k)
    return "-" if v is None else (f"{v:g}" if isinstance(v, float) else str(v))


def main():
    conds = [("v1 clean", "v1.clean"), ("v1 + renderings", "v1.rend"), ("v2 clean", "v2.clean"), ("v2 + renderings", "v2.rend")]
    print("**Table 35.1: the embedding route (all-MiniLM-L6-v2, contrastive, 96 trained and 24 held-out merchants): 12-way category by nearest category text (accuracy %; chance 8.3)**\n")
    print("| test string | " + " | ".join(c[0] for c in conds) + " |")
    print("|---|" + "---|" * len(conds))
    for label, k in (("name (trained)", "name_train"), ("bank string (trained)", "bank_train"), ("bank string, normalised (trained)", "bank_norm_train"),
                     ("truncated string (trained)", "bank_hard_train"), ("truncated, normalised (trained)", "bank_hard_norm_train"), ("description (trained)", "desc_train"),
                     ("name (held out)", "name_heldout"), ("bank string (held out)", "bank_heldout"), ("bank string, normalised (held out)", "bank_norm_heldout"),
                     ("truncated, normalised (held out)", "bank_hard_norm_heldout"), ("description (held out)", "desc_heldout"), ("training minutes", "train_minutes")):
        print(f"| {label} | " + " | ".join(g(f"embedding.{c}", k) for _, c in conds) + " |")
    print("\n**Table 35.2: the language-model route (Qwen2.5-0.5B, full fine-tuning at 1e-5, 420 steps, all 120 merchants trained): section 4's formats plus the normalised and truncated bank strings, and the products-to-category bridge (accuracy %; chance 8.3 on 12-way, 25 on 4-way)**\n")
    print("| measure | " + " | ".join(c[0] for c in conds) + " |")
    print("|---|" + "---|" * len(conds))
    for label, k in (("clean_category (12-way)", "clean_category"), ("bank_category", "bank_category"), ("bank, normalised", "bank_norm_category"),
                     ("truncated bank", "bank_hard_category"), ("truncated, normalised", "bank_hard_norm_category"), ("sells (4-way)", "sells"), ("reverse (4-way)", "reverse"),
                     ("P(clean_category | sells correct)", "clean_category_given_sells"), ("merchants with sells correct", "n_sells_correct"),
                     ("clean_category, single-category merchants", "clean_category_single"), ("clean_category, multi-category merchants", "clean_category_multi"),
                     ("sells, single / multi", "sells_single"), ("sells, multi", "sells_multi"), ("perplexity, neutral English", "ppl_general"), ("training texts", "n_texts"), ("training minutes", "train_minutes")):
        print(f"| {label} | " + " | ".join(g(f"llm.{c}", k) for _, c in conds) + " |")


if __name__ == "__main__":
    main()
