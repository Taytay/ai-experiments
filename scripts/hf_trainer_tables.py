"""Tables for REPORT.md section 44 (PLAN step 38, INFRA-2): the categoriser trained through transformers + peft (TRAINER=hf) against
the unsloth trainer of section 38, same seed, same batches; and the two scorers (unsloth / plain transformers + peft) on both adapters.

  44.1  REAL-6 accuracy per arm (no DB, record in prompt), training minutes, peak memory, final loss, trainable parameters
  44.2  the training loss at every 25 steps, both trainers (from the tracker's train_loss@N points)
  44.3  scorer agreement: each adapter under both scorers, accuracy and the share of identical predictions
  44.4  the 4-bit default measured: the instruct base on REAL-6, the Qwen2.5-3B base and arm C on the section 33 items, graph4 and the
        LRE probe, read on the 4-bit base (as the report had them) and on the bf16 base (RUN_TAG=bf16 / SCORER=hf re-reads)
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
          "(section 38: QLoRA on the NF4 4-bit base, scored on it) and through transformers + peft alone (TRAINER=hf: bf16 base, scored on it with SCORER=hf); accuracy %, "
          "DB-only groups from the per-item files**\n")
    cols = [(f"{a}: unsloth", db, cond, f"{CAT}_{db}_lora", f"categoriser_llm_{db}.json") for a, db, cond in arms] + \
           [(f"{a}: transformers + peft", db, cond, f"{CAT}_{db}_hf_lora_hfs", f"categoriser_llm_{db}_hf.json") for a, db, cond in arms]
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
    print("\n**Table 44.3: the two scorers on each adapter (the same option log-probability rule): unsloth's loader puts every adapter on the 4-bit base it defaults to; "
          "transformers + peft loads the base named in the adapter's config, the 4-bit one for unsloth's adapters and bf16 for its own. Accuracy and the share of items with the same prediction**\n")
    rows = [("no DB, unsloth adapter (4-bit)", f"{CAT}_none_lora", f"{CAT}_none_lora_hfs", "noctx"),
            ("no DB, transformers + peft adapter (bf16)", f"{CAT}_none_hf_lora", f"{CAT}_none_hf_lora_hfs", "noctx"),
            ("record in prompt, transformers + peft adapter (bf16)", f"{CAT}_ret_hf_lora", f"{CAT}_ret_hf_lora_hfs", "ctx")]
    print("| adapter | unsloth scorer | transformers + peft scorer | same prediction |")
    print("|---|---|---|---|")
    for label, t1, t2, cond in rows:
        a, b = per_item(t1, cond), per_item(t2, cond)
        acc1 = "-" if a is None else acc(a, lambda r: True); acc2 = "-" if b is None else acc(b, lambda r: True)
        same = "-"
        if a and b:
            pb = {r["id"]: r["pred"] for r in b}
            same = f"{100 * sum(r['pred'] == pb.get(r['id']) for r in a) / len(a):.1f}%"
        print(f"| {label} | {acc1} | {acc2} | {same} |")


def per_item_any(stem, cond):
    p = R / "per_item" / f"{stem}.{cond}.jsonl"
    return [json.loads(l) for l in open(p)] if p.exists() else None


def agree(a, b, levels=None):
    if not a or not b:
        return "-"
    pb = {r["id"]: r["pred"] for r in b}
    xs = [r["pred"] == pb.get(r["id"]) for r in a if r["id"] in pb and (levels is None or r["level"] in levels)]
    return f"{100 * sum(xs) / len(xs):.1f}%" if xs else "-"


def t444():
    C = "curriculum_Qwen2.5-3B_C_p200_lora"
    print("\n**Table 44.4: the size of the 4-bit default. The same items and scorers read on the NF4 4-bit base (the numbers the report carried) and on the bf16 base "
          "(RUN_TAG=bf16 re-reads; SCORER=hf for REAL-6). The section 33 items are 160 per attribute, the ICL suites 48 per task, ARC and MMLU 200; "
          "the LRE probe is the type relation in the trained sentence, cross-validated accuracy of the linear map (section 40)**\n")
    print("| model, measure (section) | 4-bit read | bf16 read | same prediction |")
    print("|---|---|---|---|")
    r4, rb = load(R / "real6_Qwen2.5-3B-Instruct.json"), load(R / "real6_Qwen2.5-3B-Instruct_hfs.json")
    for label, cond, key in (("Instruct base, REAL-6 no record, all (38)", "noctx", "R6_all"), ("Instruct base, REAL-6 no record, unseen merchant", "noctx", "R6_unseen_all"),
                             ("Instruct base, REAL-6 record in prompt, all (38)", "ctx", "R6_all"), ("Instruct base, REAL-6 record in prompt, unseen merchant", "ctx", "R6_unseen_all")):
        a, b = r4.get(cond, {}).get(key), rb.get(cond, {}).get(key)
        same = agree(per_item("Qwen2.5-3B-Instruct", cond), per_item("Qwen2.5-3B-Instruct_hfs", cond)) if key == "R6_all" else ""
        print(f"| {label} | {'-' if a is None else f'{a:g}'} | {'-' if b is None else f'{b:g}'} | {same} |")
    for name, stem, cond in (("Qwen2.5-3B base", "items2_base", "base"), ("arm C", f"items2_{C}", "trained")):
        d4, db = load(R / f"{stem}.json").get(cond, {}), load(R / f"{stem}_bf16.json").get(cond, {})
        p4, pb = per_item_any(stem, cond), per_item_any(f"{stem}_bf16", cond)
        for label, key, levels in ((f"{name}, v2 type induction (33)", "I2_type", ["I2_type"]), (f"{name}, v2 habitat / diet / region mean (33)", ("I2_habitat", "I2_diet", "I2_region"), ["I2_habitat", "I2_diet", "I2_region"]),
                                   (f"{name}, ICL symbol, v1 suite (33)", "ICL_symbol_mean", None), (f"{name}, ICL natural, v1 suite", "ICL_natural_mean", None),
                                   (f"{name}, ICL symbol, v2 suite", "ICL2_symbol_mean", None), (f"{name}, ICL natural, v2 suite", "ICL2_natural_mean", None),
                                   (f"{name}, ARC-Easy", "K_arc_easy", ["K_arc_easy"]), (f"{name}, MMLU 5-shot", "K_mmlu", ["K_mmlu"])):
            if isinstance(key, tuple):
                a = sum(d4[k] for k in key) / 3 if all(k in d4 for k in key) else None; b = sum(db[k] for k in key) / 3 if all(k in db for k in key) else None
            else:
                a, b = d4.get(key), db.get(key)
            if levels is None:
                pre = key.split("_")[0] + "_" + key.split("_")[1]
                levels = sorted({r["level"] for r in (p4 or []) if r["level"].startswith(pre + "_")})
            print(f"| {label} | {'-' if a is None else f'{a:.1f}'} | {'-' if b is None else f'{b:.1f}'} | {agree(p4, pb, levels)} |")
    g4, gb = load(R / f"graph4_{C}.json").get("trained", {}), load(R / f"graph4_{C}_bf16.json").get("trained", {})
    print(f"| arm C, type -> weakness, path form (42) | {g4.get('G4_type_weakness_path', '-')} | {gb.get('G4_type_weakness_path', '-')} | "
          f"{agree(per_item_any(f'graph4_{C}', 'trained'), per_item_any(f'graph4_{C}_bf16', 'trained'), ['G4_type_weakness_path'])} |")
    l4, lb = load(R / f"lre_{C}_tf.json").get("type_noctx", {}), load(R / f"lre_{C}_tf_bf16.json").get("type_noctx", {})
    for L in (20, 32, 36):
        a, b = l4.get(f"L{L}_cv_acc"), lb.get(f"L{L}_cv_acc")
        print(f"| arm C, LRE type probe, layer {L} (40) | {'-' if a is None else f'{a:.1f}'} | {'-' if b is None else f'{b:.1f}'} |  |")
    a, b = l4.get("model_acc_seen"), lb.get("model_acc_seen")
    print(f"| arm C, LRE probe, the model's own answer on the 136 species | {'-' if a is None else f'{a:.1f}'} | {'-' if b is None else f'{b:.1f}'} |  |")


def main():
    t441(); t442(); t443(); t444()


if __name__ == "__main__":
    main()
