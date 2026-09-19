"""Table 37.1 for REPORT.md section 37 (PLAN step 35, REAL-6): the REAL-6 cells (merchant seen / unseen in the user's history x
category name standard / renamed / new) for every scored model, from results/real6_<tag>.json. Each cell is `acc [ci]`, with * when
the accuracy is inside its null band (gold permuted within the cell, predictions fixed; section 11).

usage: uv run python scripts/real6_tables.py            prints markdown; '-' where a run is missing
"""
import json

from ai_experiments.paths import ROOT

R = ROOT / "results"
COLS = [("Qwen2.5-3B base, 24-shot", "Qwen2.5-3B", "noctx"), ("+ fact-DB record", "Qwen2.5-3B", "ctx"),
        ("Qwen2.5-3B-Instruct, 24-shot", "Qwen2.5-3B-Instruct", "noctx"), ("Instruct + record", "Qwen2.5-3B-Instruct", "ctx"),
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


def in_prompt_table():
    """Table 37.2: the LLM sees only 24 of the 300 history rows, so 'seen merchant' splits into merchants among the 24 shots and merchants
    elsewhere in the history; computed from the per-item files and the frozen set, no rescoring."""
    from ai_experiments import real6 as R6
    doc = R6.load()
    shot_merchants = {}
    for u in doc["users"]:
        text2m = {h["text"]: h["merchant"] for h in u["history"]}
        shot_merchants[u["user"]] = {text2m[t] for t in u["shots"] if t in text2m}
    item = {it["id"]: it for it in doc["items"]}
    print("\n**Table 37.2: the same runs by whether the merchant is among the 24 prompt shots, elsewhere in the 300-row history, or absent from it (accuracy %; "
          "the encoder 'full history' columns use all 300 rows, so their first two groups differ only in the merchant's frequency)**\n")
    print("| group | items | " + " | ".join(c[0] for c in COLS) + " |")
    print("|---|---|" + "---|" * len(COLS))
    groups = ("in the 24 shots", "in the history, not the shots", "not in the history", "known chain, not in the history", "opaque, not in the history")
    def grp(r):
        it = item[r["id"]]
        if not it["seen"]:
            return ["not in the history", ("known chain" if it["known"] else "opaque") + ", not in the history"]
        return ["in the 24 shots" if it["merchant"] in shot_merchants[it["user"]] else "in the history, not the shots"]
    counts = {}
    cells = {}
    for _, t, c in COLS:
        p = R / "per_item" / f"real6_{t}.{c}.jsonl"
        if not p.exists():
            continue
        acc = {g: [] for g in groups}
        for line in open(p):
            r = json.loads(line)
            for g in grp(r):
                acc[g].append(r["correct"])
        for g in groups:
            cells[(t, c, g)] = f"{100 * sum(acc[g]) / len(acc[g]):.1f}" if acc[g] else "-"
            counts[g] = len(acc[g])
    for g in groups:
        print(f"| {g} | {counts.get(g, '-')} | " + " | ".join(cells.get((t, c, g), "-") for _, t, c in COLS) + " |")


def main():
    print("**Table 37.1: the REAL-6 evaluation set (20 synthetic users, 8 to 20 categories each, 24-shot prompts of the user's own history) by cell: "
          "accuracy % [95% bootstrap interval], * = inside the null band; chance is the mean of 1/(categories per user), about 7**\n")
    print("| cell | items | " + " | ".join(c[0] for c in COLS) + " |")
    print("|---|---|" + "---|" * len(COLS))
    d0 = load("Qwen2.5-3B") or load("minilm")
    for label, key in ROWS:
        n = next((v[key + "_n"] for v in d0.values() if isinstance(v, dict) and key + "_n" in v), "-")
        print(f"| {label} | {n} | " + " | ".join(cell(load(t), c, key) for _, t, c in COLS) + " |")
    in_prompt_table()


if __name__ == "__main__":
    main()
