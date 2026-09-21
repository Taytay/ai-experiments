"""Tables for REPORT.md section 44 (PLAN step 38, INFRA-2): the categoriser trained through transformers + peft (TRAINER=hf) against
the unsloth trainer of section 38, same seed, same batches; and the two scorers (unsloth / plain transformers + peft) on both adapters.

  44.1  REAL-6 accuracy per arm (no DB, record in prompt), training minutes, peak memory, final loss, trainable parameters
  44.2  the training loss at every 25 steps, both trainers (from the tracker's train_loss@N points)
  44.3  scorer agreement on the no-DB adapters: accuracy under each scorer and the share of identical predictions
usage: uv run python scripts/hf_trainer_tables.py            prints markdown; '-' where a run is missing
"""
import json

from ai_experiments import real6 as R6
from ai_experiments.paths import ROOT

R = ROOT / "results"
CAT = "categoriser_Qwen2.5-3B-Instruct"
DOC = R6.load()
ITEM = {it["id"]: it for it in DOC["items"]}
DB_ONLY = R6.db_only_merchants()


def load(p):
    return json.load(open(p)) if p.exists() else {}


def per_item(tag, cond):
    p = R / "per_item" / f"real6_{tag}.{cond}.jsonl"
    return [json.loads(l) for l in open(p)] if p.exists() else None


def acc(recs, pred):
    xs = [r["correct"] for r in recs if pred(r)]
    return f"{100 * sum(xs) / len(xs):.1f}" if xs else "-"


def t441():
    arms = [("no DB", "none", "noctx"), ("record in prompt", "ret", "ctx")]
    print("**Table 44.1: the same categoriser recipe (rank-64 LoRA on every linear layer, 200 steps x 16 sequences, lr 1e-4, seed 0, identical batches) trained through unsloth "
          "(section 38) and through transformers + peft alone (TRAINER=hf), scored by the section 38 scorer (accuracy %; DB-only groups from the per-item files)**\n")
    cols = [(f"{a}: unsloth", db, cond, f"{CAT}_{db}_lora", f"categoriser_llm_{db}.json") for a, db, cond in arms] + \
           [(f"{a}: transformers + peft", db, cond, f"{CAT}_{db}_hf_lora", f"categoriser_llm_{db}_hf.json") for a, db, cond in arms]
    cols = [cols[0], cols[2], cols[1], cols[3]]
    print("| measure | " + " | ".join(c[0] for c in cols) + " |")
    print("|---|" + "---|" * len(cols))
    for label, key in (("all items", "R6_all"), ("seen merchant", "R6_seen_all"), ("unseen merchant", "R6_unseen_all"), ("unseen, new word", "R6_unseen_new")):
        cells = []
        for _, db, cond, tag, _ in cols:
            r = load(R / f"real6_{tag}.json").get(cond, {})
            cells.append("-" if key not in r else f"{r[key]:g} [{r[key + '_ci'][0]:g}, {r[key + '_ci'][1]:g}]")
        print(f"| {label} | " + " | ".join(cells) + " |")
    for label, pred in (("DB-only merchants", lambda r: not ITEM[r["id"]]["seen"] and ITEM[r["id"]]["merchant"] in DB_ONLY),
                        ("DB-only, opaque", lambda r: not ITEM[r["id"]]["seen"] and ITEM[r["id"]]["merchant"] in DB_ONLY and not ITEM[r["id"]]["known"])):
        cells = []
        for _, db, cond, tag, _ in cols:
            recs = per_item(tag, cond)
            cells.append("-" if recs is None else acc(recs, pred))
        print(f"| {label} | " + " | ".join(cells) + " |")
    for label, key, fmt in (("training minutes", "train_minutes", "{:g}"), ("peak allocated GiB", "peak_alloc_GiB", "{:g}"), ("peak reserved GiB", "peak_reserved_GiB", "{:g}"),
                            ("final loss (last 10 steps)", "final_loss", "{:g}"), ("trainable parameters", "n_trainable", "{:,}")):
        cells = []
        for _, db, cond, tag, stats in cols:
            d = load(R / stats)
            cells.append("-" if key not in d else fmt.format(d[key]))
        print(f"| {label} | " + " | ".join(cells) + " |")


def tracker_losses(db, trainer):
    """train_loss@N of the seed-0 run of this arm and trainer from evals/runs.jsonl (the unsloth runs of section 38 have no 'trainer' key)."""
    best = None
    for l in open(ROOT / "evals" / "runs.jsonl"):
        d = json.loads(l)
        c = d.get("config", {})
        if d.get("experiment") != "categoriser" or c.get("route") != "llm" or c.get("db") != db or c.get("seed", 0) != 0 or c.get("run_tag") or d.get("status") != "finished":
            continue
        if c.get("trainer", "unsloth") != trainer or c.get("real6_db", "v1") != "v1":
            continue
        m = d.get("metrics", {}).get("train", {}) if isinstance(d.get("metrics"), dict) else {}
        if m:
            best = m  # the last finished run wins
    return {int(k.split("@")[1]): v for k, v in (best or {}).items() if k.startswith("train_loss@")}


def t442():
    print("\n**Table 44.2: training loss (label tokens, mean over the step's 16 sequences) at every 25 steps, both trainers, same batches**\n")
    steps = list(range(25, 201, 25))
    print("| arm, trainer | " + " | ".join(str(s) for s in steps) + " |")
    print("|---|" + "---|" * len(steps))
    for a, db in (("no DB", "none"), ("record in prompt", "ret")):
        for trainer in ("unsloth", "hf"):
            L = tracker_losses(db, trainer)
            print(f"| {a}, {'transformers + peft' if trainer == 'hf' else 'unsloth'} | " + " | ".join("-" if s not in L else f"{L[s]:.3f}" for s in steps) + " |")


def t443():
    print("\n**Table 44.3: the two scorers (unsloth's loader and plain transformers + peft; the same option log-probability rule) on the two no-DB adapters: accuracy and the share of items with the same prediction**\n")
    rows = [("unsloth adapter", f"{CAT}_none_lora", f"{CAT}_none_lora_hfs"), ("transformers + peft adapter", f"{CAT}_none_hf_lora", f"{CAT}_none_hf_lora_hfs")]
    print("| adapter | unsloth scorer | transformers + peft scorer | same prediction |")
    print("|---|---|---|---|")
    for label, t1, t2 in rows:
        a, b = per_item(t1, "noctx"), per_item(t2, "noctx")
        acc1 = "-" if a is None else acc(a, lambda r: True); acc2 = "-" if b is None else acc(b, lambda r: True)
        same = "-"
        if a and b:
            pb = {r["id"]: r["pred"] for r in b}
            same = f"{100 * sum(r['pred'] == pb.get(r['id']) for r in a) / len(a):.1f}%"
        print(f"| {label} | {acc1} | {acc2} | {same} |")


def main():
    t441(); t442(); t443()


if __name__ == "__main__":
    main()
