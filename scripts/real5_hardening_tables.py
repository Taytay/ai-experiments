"""Tables for REPORT.md section 43 (PLAN step 37, REAL-7): the section 38 record-in-prompt result without its three conveniences.

  43.1  the ambiguous fact DB (real6_v1_ambdb.json) against the disjoint one, per cell and on the DB-only merchants, for the record
        in the prompt (Instruct base, SFT trained on either DB) and injected into the weights (400-step parametric arm), plus the
        record-in-prompt accuracy by how ambiguous the record is (products off the category's pool; multi-category merchants)
  43.2  the learned retriever: recall@1 / @5 from the raw and normalised test strings, zero-shot and tuned, at 240 records and with
        5,000 decoys (results/real6_retrieval.json)
  43.3  end to end: the oracle record against the retrieved top-1 record, and the accuracy on retrieval hits and misses
  43.4  three seeds of the SFT arms: per-seed values and mean +- sd of the headline cells
usage: uv run python scripts/real5_hardening_tables.py            prints markdown; '-' where a run is missing
"""
import json
import statistics
from collections import defaultdict

from ai_experiments import real6 as R6
from ai_experiments.paths import ROOT

R = ROOT / "results"
CAT = "categoriser_Qwen2.5-3B-Instruct"
DOC = R6.load()
ITEM = {it["id"]: it for it in DOC["items"]}
DB_ONLY = R6.db_only_merchants()
AMB = json.loads(R6.AMB_PATH.read_text())["meta"] if R6.AMB_PATH.exists() else {}
ROWS = [("seen merchant, all", "R6_seen_all"), ("unseen merchant, all", "R6_unseen_all"), ("unseen, standard name", "R6_unseen_standard"), ("unseen, renamed", "R6_unseen_renamed"),
        ("unseen, new word", "R6_unseen_new"), ("all items", "R6_all")]
GROUPS = ["not in the history, labelled by other users in training", "DB-only (no user labelled it in training)", "DB-only, known chain", "DB-only, opaque",
          "DB-only, standard name", "DB-only, renamed", "DB-only, new word"]


def load(tag):
    p = R / f"real6_{tag}.json"
    return json.load(open(p)) if p.exists() else {}


def cell(tag, cond, key):
    r = load(tag).get(cond, {})
    if key not in r:
        return "-"
    lo, hi = r[key + "_ci"]; nlo, nhi = r[key + "_null"]
    return f"{r[key]:g}{'*' if nlo <= r[key] <= nhi else ''} [{lo:g}, {hi:g}]"


def per_item(tag, cond):
    p = R / "per_item" / f"real6_{tag}.{cond}.jsonl"
    return [json.loads(l) for l in open(p)] if p.exists() else None


def groups_of(r):
    it = ITEM[r["id"]]
    if it["seen"]:
        return []
    if it["merchant"] not in DB_ONLY:
        return [GROUPS[0]]
    return [GROUPS[1], "DB-only, known chain" if it["known"] else "DB-only, opaque", {"standard": "DB-only, standard name", "renamed": "DB-only, renamed", "new": "DB-only, new word"}[it["name_type"]]]


def acc(recs, pred=lambda r: True):
    xs = [r["correct"] for r in recs if pred(r)]
    return f"{100 * sum(xs) / len(xs):.1f}" if xs else "-"


def group_rows(cols):
    """The unseen-merchant split of Table 38.2 for the given (label, tag, cond) columns."""
    out = defaultdict(dict)
    for label, tag, cond in cols:
        recs = per_item(tag, cond)
        if recs is None:
            continue
        by = defaultdict(list)
        for r in recs:
            for g in groups_of(r):
                by[g].append(r)
        for g in GROUPS:
            out[g][label] = acc(by[g])
    return out


def table(title, cols, rows, extra_rows=None):
    print(f"\n**{title}**\n" if title else "")
    print("| | " + " | ".join(c[0] for c in cols) + " |")
    print("|---|" + "---|" * len(cols))
    for label, key in rows:
        print(f"| {label} | " + " | ".join(cell(t, c, key) for _, t, c in cols) + " |")
    if extra_rows:
        g = group_rows(cols)
        for name in GROUPS:
            print(f"| {name} | " + " | ".join(g[name].get(c[0], "-") for c in cols) + " |")


def t431():
    cols = [("Instruct + record, disjoint DB (sec. 37)", "Qwen2.5-3B-Instruct", "ctx"), ("Instruct + record, ambiguous DB", "Qwen2.5-3B-Instruct_amb", "ctx"),
            ("SFT + record: trained and scored disjoint (sec. 38)", f"{CAT}_ret_lora", "ctx"), ("SFT + record: trained disjoint, scored ambiguous", f"{CAT}_ret_lora_amb", "ctx"),
            ("SFT + record: trained and scored ambiguous", f"{CAT}_ret_amb_lora_amb", "ctx"),
            ("bge tuned + record on query, disjoint (sec. 38)", "categoriser_bge_none_ctx", "full"), ("bge tuned + record on query, ambiguous", "categoriser_bge_none_ctx_amb", "full"),
            ("SFT + parametric DB, 400 steps at 50%, disjoint (sec. 38)", f"{CAT}_param_x2_lora", "noctx"), ("SFT + parametric DB, 400 steps at 50%, ambiguous", f"{CAT}_param_x2_amb_lora", "noctx")]
    table("Table 43.1: the disjoint fact DB of sections 37 and 38 against the ambiguous one (each category's pool gains two products of the next category; a fifth of the merchants "
          "sell one product of another category), same items, same users (accuracy % [95% bootstrap interval], * = inside the null band; chance about 7; the unseen split follows Table 38.2)", cols, ROWS, extra_rows=True)
    # by record ambiguity, record-in-prompt conditions on the ambiguous DB
    cols2 = [c for c in cols if c[1].endswith("_amb")]
    print("\n**Table 43.1b: accuracy on the ambiguous DB by the record's ambiguity (unseen merchants only): products off the category's own pool 0 / 1 / 2, and the multi-category merchants "
          "(two own products and one of another category), against the same merchants under the disjoint DB where the columns exist**\n")
    strata = [("0 off-pool products", lambda m: m["off_pool"] == 0 and not m["multi"]), ("1 off-pool product, not multi", lambda m: m["off_pool"] == 1 and not m["multi"]),
              ("2 off-pool products", lambda m: m["off_pool"] == 2), ("multi-category merchant", lambda m: m["multi"])]
    print("| record stratum (n unseen items) | " + " | ".join(c[0] for c in cols2) + " |")
    print("|---|" + "---|" * len(cols2))
    for label, pred in strata:
        n = sum(1 for it in DOC["items"] if not it["seen"] and AMB and pred(AMB[it["merchant"]]))
        cells = []
        for _, tag, cond in cols2:
            recs = per_item(tag, cond)
            cells.append("-" if recs is None else acc(recs, lambda r: not ITEM[r["id"]]["seen"] and pred(AMB[ITEM[r["id"]]["merchant"]])))
        print(f"| {label} ({n}) | " + " | ".join(cells) + " |")


def t432():
    p = R / "real6_retrieval.json"
    if not p.exists():
        print("\n(no retrieval results)"); return
    d = json.load(open(p))
    print(f"\n**Table 43.2: the record retriever (all-MiniLM-L6-v2 over the merchant records; tuned on {d['n_pairs']} (templated rendering -> record) pairs from the DB's own names, no user labels): "
          f"recall@1 / recall@5 of the merchant's own record from the {d['config']['n_items']} test strings, raw and through the normaliser, at 240 records and with {d.get('n_decoys', 0):,} opaque decoy records added**\n")
    conds = [("zero-shot, 240 records", "zero_shot"), ("tuned, 240 records", "tuned"), (f"zero-shot, +{d.get('n_decoys', 0):,} decoys", "zero_shot_decoys"), (f"tuned, +{d.get('n_decoys', 0):,} decoys", "tuned_decoys")]
    groups = [("all strings", "all"), ("known chain", "known"), ("opaque merchant", "opaque"), ("full name in the string", "full_name"), ("name truncated or vowel-stripped", "truncated_or_abbr"), ("DB-only merchants", "db_only")]
    for q in ("raw", "norm"):
        print(f"\n*{'raw statement string' if q == 'raw' else 'normalised string'} as the query*\n")
        print("| strings (n) | " + " | ".join(c[0] for c in conds) + " |")
        print("|---|" + "---|" * len(conds))
        for label, g in groups:
            n = d["tuned"].get(f"{q}_{g}_n", "")
            print(f"| {label} ({n}) | " + " | ".join(f"{d.get(c, {}).get(f'{q}_{g}_recall@1', '-')} / {d.get(c, {}).get(f'{q}_{g}_recall@5', '-')}" for _, c in conds) + " |")


def t433():
    pairs = [("Instruct base", "Qwen2.5-3B-Instruct", "Qwen2.5-3B-Instruct"), ("SFT + record (trained on the oracle record)", f"{CAT}_ret_lora", f"{CAT}_ret_lora"),
             ("SFT + record, ambiguous DB", f"{CAT}_ret_amb_lora_amb", f"{CAT}_ret_amb_lora_amb")]
    print("\n**Table 43.3: the oracle record (found by the merchant's name) against the retrieved top-1 record (found from the statement string, right or wrong) in the prompt, "
          "and the accuracy on the items whose retrieval hit and missed (accuracy %; the bge column appends the record to the query)**\n")
    cols = []
    for label, t_or, t_ret in pairs:
        cols += [(f"{label}: oracle", t_or, "ctx"), (f"{label}: retrieved", t_ret, "ret1")]
    cols += [("bge tuned: oracle on query", "categoriser_bge_none_ctx", "full"), ("bge tuned: retrieved on query", "categoriser_bge_none_ret1", "full")]
    table("", cols, ROWS, extra_rows=True)
    print("\n| retrieval outcome (n) | " + " | ".join(c[0] for c in cols if c[2] == "ret1") + " |")
    print("|---|" + "---|" * sum(c[2] == "ret1" for c in cols))
    hits = {k: v["hit1"] for k, v in json.loads((R / "real6_retrieved.json").read_text())["items"].items()} if (R / "real6_retrieved.json").exists() else {}
    for label, want in (("retriever hit", True), ("retriever missed", False)):
        n = sum(1 for v in hits.values() if v == want)
        print(f"| {label} ({n}) | " + " | ".join("-" if per_item(t, c) is None else acc(per_item(t, c), lambda r: hits.get(r['id']) == want) for _, t, c in cols if c == "ret1") + " |")


def t434():
    arms = [("SFT, no DB", "none", "noctx"), ("SFT + record in prompt", "ret", "ctx"), ("SFT + parametric DB, 400 steps at 50%", "param_x2", "noctx")]
    keys = [("all items", "R6_all"), ("seen merchant", "R6_seen_all"), ("unseen merchant", "R6_unseen_all"), ("unseen, new word", "R6_unseen_new")]
    print("\n**Table 43.4: three seeds (LoRA init and data order) of the section 38 SFT arms: per-seed accuracy and mean +- sd (accuracy %; the DB-only groups from the per-item files)**\n")
    print("| arm | measure | seed 0 | seed 1 | seed 2 | mean +- sd |")
    print("|---|---|---|---|---|---|")
    for label, db, cond in arms:
        tags = [f"{CAT}_{db}_lora", f"{CAT}_{db}_s1_lora", f"{CAT}_{db}_s2_lora"]
        for klabel, key in keys:
            vals = [load(t).get(cond, {}).get(key) for t in tags]
            have = [v for v in vals if v is not None]
            ms = f"{statistics.mean(have):.1f} +- {statistics.stdev(have):.1f}" if len(have) > 1 else "-"
            print(f"| {label} | {klabel} | " + " | ".join("-" if v is None else f"{v:g}" for v in vals) + f" | {ms} |")
        for g in (GROUPS[1], "DB-only, opaque", GROUPS[0]):
            vals = []
            for t in tags:
                recs = per_item(t, cond)
                vals.append(None if recs is None else float(acc([r for r in recs if g in groups_of(r)])))
            have = [v for v in vals if v is not None]
            ms = f"{statistics.mean(have):.1f} +- {statistics.stdev(have):.1f}" if len(have) > 1 else "-"
            print(f"| {label} | {g} | " + " | ".join("-" if v is None else f"{v:.1f}" for v in vals) + f" | {ms} |")
    for db in ("none", "ret", "param_x2"):
        mins = []
        for sfx in ("", "_s1", "_s2"):
            p = R / f"categoriser_llm_{db}{sfx}.json"
            mins.append(str(json.load(open(p)).get("train_minutes")) if p.exists() else "-")
        print(f"\n{db}: training minutes per seed {', '.join(mins)}")


def main():
    t431(); t432(); t433(); t434()


if __name__ == "__main__":
    main()
