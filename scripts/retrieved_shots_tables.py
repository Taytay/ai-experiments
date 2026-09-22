"""Tables for REPORT.md section 47 (PLAN step 41, REAL-9): the 24 shots chosen per query by a rule (ai_experiments.real6_shots) instead of
the frozen stratified block, on the LLM arms.

  47.1  accuracy by rule (fixed, recent, nearest, transact, cluster) for the untrained base, the fixed-shot adapters read with the
        rule's shots at test, and the adapters trained with the rule's shots; all items, seen, unseen, DB-only, with the interval
  47.2  paired with the fixed block per item: share of the same predictions, both right, either right, net gain
  47.3  the seen items by whether the query's merchant is among the shots, and all items by whether the gold label is among the shot labels
  47.4  training minutes and final loss of the rule-trained adapters
  47.5  the corrected cells (ai_experiments.real6_cells, QUESTIONS.md REAL-13): in the history / labelled seen but not in it /
        determined by the merchant's standard category / split category; the model with the nearest-row lookup in front
        (hybrid); all items with a user-resampled interval
usage: uv run python scripts/retrieved_shots_tables.py            prints markdown; '-' where a run is missing
"""
import json
import sys
from pathlib import Path

from ai_experiments.paths import ROOT

sys.path.insert(0, str(Path(__file__).parent))
import categoriser_tables as CT  # noqa: E402

R = ROOT / "results"
CAT = "categoriser_Qwen2.5-3B-Instruct"
RULES = ["fixed", "recent", "nearest", "transact", "cluster"]
# (arm label, tag for the fixed block, condition, tag pattern for a rule)
ARMS = [("Instruct base, no record", "Qwen2.5-3B-Instruct", "noctx", "Qwen2.5-3B-Instruct_shots{r}"),
        ("Instruct base + record", "Qwen2.5-3B-Instruct", "ctx", "Qwen2.5-3B-Instruct_shots{r}"),
        ("SFT no DB (fixed-shot adapter), rule shots at test", f"{CAT}_none_lora", "noctx", f"{CAT}_none_lora_shots{{r}}"),
        ("SFT + record (fixed-shot adapter), rule shots at test", f"{CAT}_ret_lora", "ctx", f"{CAT}_ret_lora_shots{{r}}"),
        ("SFT no DB, trained and read with the rule", f"{CAT}_none_lora", "noctx", f"{CAT}_none_shots{{r}}_lora"),
        ("SFT + record, trained and read with the rule", f"{CAT}_ret_lora", "ctx", f"{CAT}_ret_shots{{r}}_lora")]
KEYS = [("all items", "R6_all"), ("seen merchant", "R6_seen_all"), ("unseen merchant", "R6_unseen_all"), ("unseen, renamed", "R6_unseen_renamed"), ("unseen, new word", "R6_unseen_new")]


def tag_of(arm, rule):
    return arm[1] if rule == "fixed" else arm[3].format(r=rule)


FROZEN_FLAGS = None


def recs(tag, cond):
    """Per-item records; the fixed-block runs predate the shot flags, so those are filled from the frozen set (real6_shots.frozen_flags)."""
    global FROZEN_FLAGS
    p = R / "per_item" / f"real6_{tag}.{cond}.jsonl"
    out = {r["id"]: r for r in map(json.loads, open(p))} if p.exists() else {}
    if out and "merchant_in_shots" not in next(iter(out.values())):
        if FROZEN_FLAGS is None:
            from ai_experiments import real6 as R6
            from ai_experiments.real6_shots import frozen_flags
            doc = R6.load(); users = {u["user"]: u for u in doc["users"]}
            FROZEN_FLAGS = {it["id"]: {k: v for k, v in frozen_flags(it, users[it["user"]]).items() if k in ("merchant_in_shots", "gold_in_shots")} for it in doc["items"]}
        for i, r in out.items():
            r.update(FROZEN_FLAGS[i])
    return out


def db_only_acc(rs):
    from ai_experiments import real6 as R6
    db = R6.db_only_merchants()
    x = [r["correct"] for r in rs.values() if r["merchant"] in db and r["level"].startswith("R6_unseen")]
    return f"{100 * sum(x) / len(x):.1f}" if x else "-"


def t1():
    print("**Table 47.1: the shot rules on the REAL-6 items (accuracy % [95% bootstrap interval]; the fixed column is the frozen stratified block of sections 37, 38, 43; "
          "4-bit throughout; DB-only = unseen merchants no user labelled in training)**\n")
    print("| arm | cell | " + " | ".join(RULES) + " |")
    print("|---|---|" + "---|" * len(RULES))
    for arm in ARMS:
        for label, key in KEYS + [("DB-only merchants", "db_only")]:
            cells = []
            for rule in RULES:
                d = CT.load(tag_of(arm, rule))
                if key == "db_only":
                    cells.append(db_only_acc(recs(tag_of(arm, rule), arm[2])))
                else:
                    cells.append(CT.cell(d, arm[2], key))
            print(f"| {arm[0]} | {label} | " + " | ".join(cells) + " |")


def t2():
    print("\n**Table 47.2: each rule paired per item with the fixed block for the same arm (share of items with the same prediction; both right; either right; net gain in points)**\n")
    print("| arm | " + " | ".join(RULES[1:]) + " |")
    print("|---|" + "---|" * (len(RULES) - 1))
    for arm in ARMS:
        base = recs(arm[1], arm[2]); cells = []
        for rule in RULES[1:]:
            rs = recs(tag_of(arm, rule), arm[2])
            if not base or not rs or set(base) != set(rs):
                cells.append("-"); continue
            n = len(base)
            same = sum(base[i]["pred"] == rs[i]["pred"] for i in base) / n; both = sum(base[i]["correct"] and rs[i]["correct"] for i in base) / n
            either = sum(base[i]["correct"] or rs[i]["correct"] for i in base) / n; gain = (sum(r["correct"] for r in rs.values()) - sum(r["correct"] for r in base.values())) / n
            cells.append(f"same {100 * same:.1f}, both {100 * both:.1f}, either {100 * either:.1f}, {100 * gain:+.1f}")
        print(f"| {arm[0]} | " + " | ".join(cells) + " |")


def t3():
    print("\n**Table 47.3: seen items by whether the query's merchant is among the 24 shots (accuracy %, share of items in the group), and all items by whether the gold label "
          "is among the shot labels**\n")
    print("| arm | group | " + " | ".join(RULES) + " |")
    print("|---|---|" + "---|" * len(RULES))
    groups = (("seen, merchant among the shots", lambda r: r["level"].startswith("R6_seen") and r.get("merchant_in_shots")),
              ("seen, merchant not among the shots", lambda r: r["level"].startswith("R6_seen") and r.get("merchant_in_shots") is False),
              ("gold label among the shot labels", lambda r: r.get("gold_in_shots")), ("gold label not among the shot labels", lambda r: r.get("gold_in_shots") is False))
    for arm in ARMS:
        for g, f in groups:
            cells = []
            for rule in RULES:
                rs = recs(tag_of(arm, rule), arm[2])
                x = [r["correct"] for r in rs.values() if f(r)]
                cells.append(f"{100 * sum(x) / len(x):.1f} ({100 * len(x) / len(rs):.0f}%)" if x else "-")
            print(f"| {arm[0]} | {g} | " + " | ".join(cells) + " |")


def t4():
    print("\n**Table 47.4: training cost of the rule-trained adapters (minutes, final loss over the last steps)**\n")
    print("| adapter | " + " | ".join(RULES) + " |")
    print("|---|" + "---|" * len(RULES))
    for db in ("none", "ret"):
        cells = []
        for rule in RULES:
            p = R / f"categoriser_llm_{db}{'' if rule == 'fixed' else '_shots' + rule}.json"
            d = json.load(open(p)) if p.exists() else {}
            cells.append(f"{d.get('train_minutes', '-')} min, loss {d.get('final_loss', '-')}" if d else "-")
        print(f"| SFT {db} | " + " | ".join(cells) + " |")


def t5():
    from ai_experiments import real6_cells as RC
    ks = RC.KINDS[:4]
    print("\n**Table 47.5: the corrected cells (accuracy %; groups from `real6_cells`: 444 items whose merchant is in the user's history, 115 labelled seen "
          "whose merchant the 300-row cut removed, 509 determined by the merchant's standard category, 103 in a category the user split; 'hybrid' = the label of "
          "the nearest history row when its retriever cosine is at least 0.8, else the model; interval = users resampled)**\n")
    print("| arm | rule | " + " | ".join(ks) + " | all [user interval] | hybrid, all |")
    print("|---|---|" + "---|" * (len(ks) + 2))
    nn = RC.nn1()
    print("| nearest-row lookup alone | - | " + " | ".join(f"{100 * sum(nn[i][1] for i in nn if RC.kind(i) == k) / sum(RC.kind(i) == k for i in nn):.1f}" for k in ks)
          + f" | {100 * sum(v[1] for v in nn.values()) / len(nn):.1f} | - |")
    for arm in ARMS:
        for rule in RULES:
            rs = recs(tag_of(arm, rule), arm[2])
            if not rs:
                continue
            cells = []
            for k in ks:
                x = [r["correct"] for i, r in rs.items() if RC.kind(i) == k]
                cells.append(f"{100 * sum(x) / len(x):.1f}" if x else "-")
            c = {i: r["correct"] for i, r in rs.items()}; lo, hi = RC.user_ci(c); h = RC.hybrid(rs)
            print(f"| {arm[0]} | {rule} | " + " | ".join(cells) + f" | {100 * sum(c.values()) / len(c):.1f} [{lo}, {hi}] | {100 * sum(h.values()) / len(h):.1f} |")


if __name__ == "__main__":
    t1(); t2(); t3(); t4(); t5()
