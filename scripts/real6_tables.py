"""Table 37.1 for REPORT.md section 37 (PLAN step 35, REAL-6): the REAL-6 cells (merchant seen / unseen in the user's history x
category name standard / renamed / new) for every scored model, from results/real6_<tag>.json. Each cell is `acc [ci]`, with * when
the accuracy is inside its null band (gold permuted within the cell, predictions fixed; section 11).

usage: uv run python scripts/real6_tables.py            prints markdown; '-' where a run is missing
"""
import json

from ai_experiments.paths import ROOT

R = ROOT / "results"
COLS = [("Qwen2.5-3B base, 24-shot", "Qwen2.5-3B", "noctx"), ("+ fact-DB record", "Qwen2.5-3B", "ctx"),
        ("MiniLM prototype, 24 shots", "minilm", "shots"), ("MiniLM prototype, full history", "minilm", "full"), ("MiniLM mix (name + history)", "minilm", "mix"),
        ("bge-base prototype, full history", "bge", "full"), ("bge-base mix", "bge", "mix")]
ROWS = [("seen merchant, standard name", "R6_seen_standard"), ("seen merchant, renamed", "R6_seen_renamed"), ("seen merchant, new word", "R6_seen_new"),
        ("unseen merchant, standard name", "R6_unseen_standard"), ("unseen merchant, renamed", "R6_unseen_renamed"), ("unseen merchant, new word", "R6_unseen_new"),
        ("seen merchant, all", "R6_seen_all"), ("unseen merchant, all", "R6_unseen_all"), ("standard names, all", "R6_all_standard"), ("renamed, all", "R6_all_renamed"),
        ("new words, all", "R6_all_new"), ("all items", "R6_all")]


def load(tag):
    p = R / f"real6_{tag}.json"
    return json.load(open(p)) if p.exists() else {}


def cell(d, cond, key):
    r = d.get(cond, {})
    if key not in r:
        return "-"
    lo, hi = r[key + "_ci"]; nlo, nhi = r[key + "_null"]
    star = "*" if nlo <= r[key] <= nhi else ""
    return f"{r[key]:g}{star} [{lo:g}, {hi:g}]"


def main():
    print("**Table 37.1: the REAL-6 evaluation set (20 synthetic users, 8 to 20 categories each, 24-shot prompts of the user's own history) by cell: "
          "accuracy % [95% bootstrap interval], * = inside the null band; chance is the mean of 1/(categories per user), about 7**\n")
    print("| cell | items | " + " | ".join(c[0] for c in COLS) + " |")
    print("|---|---|" + "---|" * len(COLS))
    d0 = load("Qwen2.5-3B") or load("minilm")
    for label, key in ROWS:
        n = next((v[key + "_n"] for v in d0.values() if isinstance(v, dict) and key + "_n" in v), "-")
        print(f"| {label} | {n} | " + " | ".join(cell(load(t), c, key) for _, t, c in COLS) + " |")


if __name__ == "__main__":
    main()
