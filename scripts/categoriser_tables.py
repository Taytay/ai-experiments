"""Tables for REPORT.md section 38 (PLAN step 33, REAL-5): the trained categorisers on the REAL-6 cells beside the section 37 untrained
columns, and their general-ability check (ARC-Easy, MMLU from exp_items_v2). Table 38.1 = the cells (acc [ci], * inside the null band);
Table 38.2 = the merchant-in-shots split; Table 38.3 = ARC-Easy and MMLU of the LLM adapters against the instruct base.

usage: uv run python scripts/categoriser_tables.py            prints markdown; '-' where a run is missing
"""
import json

from ai_experiments.paths import ROOT

R = ROOT / "results"
# (label, results tag, condition)
COLS = [("Instruct, 24-shot (sec. 37)", "Qwen2.5-3B-Instruct", "noctx"), ("Instruct + record (sec. 37)", "Qwen2.5-3B-Instruct", "ctx"),
        ("SFT, no DB", "categoriser_Qwen2.5-3B-Instruct_none_lora", "noctx"), ("SFT + parametric DB", "categoriser_Qwen2.5-3B-Instruct_param_lora", "noctx"),
        ("SFT + parametric DB, 400 steps at 50%", "categoriser_Qwen2.5-3B-Instruct_param_x2_lora", "noctx"), ("SFT + record in prompt", "categoriser_Qwen2.5-3B-Instruct_ret_lora", "ctx"),
        ("bge frozen, full history (sec. 37)", "bge", "full"), ("bge frozen, mix (sec. 37)", "bge", "mix"),
        ("bge tuned, no DB", "categoriser_bge_none", "full"), ("bge tuned, mix", "categoriser_bge_none", "mix"), ("bge tuned + record on query", "categoriser_bge_none_ctx", "full"),
        ("bge tuned + parametric DB", "categoriser_bge_param", "full")]
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
    return f"{r[key]:g}{'*' if nlo <= r[key] <= nhi else ''} [{lo:g}, {hi:g}]"


def in_shots():
    from ai_experiments import real6 as R6
    doc = R6.load()
    shot_m = {}
    for u in doc["users"]:
        t2m = {h["text"]: h["merchant"] for h in u["history"]}
        shot_m[u["user"]] = {t2m[t] for t in u["shots"] if t in t2m}
    item = {it["id"]: it for it in doc["items"]}
    db_only = R6.db_only_merchants()
    groups = ("in the 24 shots", "in the history, not the shots", "not in the history", "not in the history, labelled by other users in training",
              "not in the history, DB-only (no user labelled it in training)", "DB-only, known chain", "DB-only, opaque", "DB-only, standard name", "DB-only, renamed", "DB-only, new word")
    def grp(r):
        it = item[r["id"]]
        if not it["seen"]:
            g = ["not in the history"]
            if it["merchant"] in db_only:
                g += ["not in the history, DB-only (no user labelled it in training)", "DB-only, known chain" if it["known"] else "DB-only, opaque",
                      {"standard": "DB-only, standard name", "renamed": "DB-only, renamed", "new": "DB-only, new word"}[it["name_type"]]]
            else:
                g.append("not in the history, labelled by other users in training")
            return g
        return ["in the 24 shots" if it["merchant"] in shot_m[it["user"]] else "in the history, not the shots"]
    print("\n**Table 38.2: by whether the merchant is among the 24 prompt shots, elsewhere in the 300-row history, or absent from it, and for the absent ones whether "
          "another user's training rows carried it or only the fact DB knows it (accuracy %; the untrained section 37 columns did not train, so their split is a merchant subset only)**\n")
    print("| group | " + " | ".join(c[0] for c in COLS) + " |")
    print("|---|" + "---|" * len(COLS))
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
    for g in groups:
        print(f"| {g} | " + " | ".join(cells.get((t, c, g), "-") for _, t, c in COLS) + " |")


def general():
    print("\n**Table 38.3: general ability of the SFT adapters (exp_items_v2: ARC-Easy and 5-shot MMLU, 200 items each; the v1 and v2 ICL suites)**\n")
    cols = [("Instruct base", "base_Qwen2.5-3B-Instruct", "base"), ("SFT, no DB", "categoriser_Qwen2.5-3B-Instruct_none_lora", "trained"),
            ("SFT + parametric DB", "categoriser_Qwen2.5-3B-Instruct_param_lora", "trained"), ("SFT + parametric DB, 400 steps at 50%", "categoriser_Qwen2.5-3B-Instruct_param_x2_lora", "trained"),
            ("SFT + record in prompt", "categoriser_Qwen2.5-3B-Instruct_ret_lora", "trained")]
    print("| measure | " + " | ".join(c[0] for c in cols) + " |")
    print("|---|" + "---|" * len(cols))
    for label, key in (("ARC-Easy", "K_arc_easy"), ("MMLU 5-shot", "K_mmlu"), ("ICL symbol (v1 suite)", "ICL_symbol_mean"), ("ICL natural (v1 suite)", "ICL_natural_mean"),
                       ("ICL symbol (v2 suite)", "ICL2_symbol_mean"), ("ICL natural (v2 suite)", "ICL2_natural_mean")):
        cells = []
        for _, t, c in cols:
            p = R / f"items2_{t}.json"
            v = json.load(open(p)).get(c, {}).get(key) if p.exists() else None
            cells.append("-" if v is None else f"{v:g}")
        print(f"| {label} | " + " | ".join(cells) + " |")
    for db in ("none", "param", "param_x2", "ret"):
        p = R / f"categoriser_llm_{db}.json"
        if p.exists():
            d = json.load(open(p)); print(f"\nSFT {db}: {d.get('train_minutes')} minutes, final loss {d.get('final_loss')}, {d.get('seqs_sft')} history sequences and {d.get('seqs_db')} DB sequences.")


def main():
    print("**Table 38.1: the trained categorisers on the REAL-6 cells beside the untrained columns of section 37 (accuracy % [95% bootstrap interval], * = inside the null band; chance about 7)**\n")
    print("| cell | " + " | ".join(c[0] for c in COLS) + " |")
    print("|---|" + "---|" * len(COLS))
    for label, key in ROWS:
        print(f"| {label} | " + " | ".join(cell(load(t), c, key) for _, t, c in COLS) + " |")
    in_shots()
    general()


if __name__ == "__main__":
    main()
