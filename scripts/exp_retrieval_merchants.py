"""End-to-end retrieval on the merchant set (PLAN step 14, REAL-2): recall@k of the merchant's own record from its bank
string, then the LLM's category accuracy with the retrieved record in the prompt (right or wrong), against the oracle
record and no context.

usage: uv run python scripts/exp_retrieval_merchants.py [model ...]     default: Qwen/Qwen2.5-0.5B Qwen/Qwen2.5-3B
       SMOKE=1 scores 24 items per format and writes *_smoke files.

Retrievers over the 120 canonical records (`merchants.raw_fact`), all-MiniLM-L6-v2: zero-shot; category-tuned (the section 4
pairs, sentences and names of the 96 training merchants -> category text, 6 epochs); record-tuned (the same anchors -> the
merchant's own record, the retrieval task itself). Queries: bank strings and bare names; a query's gold document is its own
merchant's record. The LLM conditions use the record-tuned retriever.

LLM conditions on the section 4 formats `bank_category` and `clean_category` (12-way, chance 8.3), untrained base models:
  none        the format's few-shot prompt as in section 4
  oracle      the merchant's own record prepended before the final query (section 4's `incontext`)
  ret1        the retriever's top-1 record from the bank string (or the name for clean_category), right or wrong
  ret3        the top-3 records, in retrieval order
Held-out merchants (2 per category, never in the retriever's training pairs) are reported separately.
"""
import json
import os
import random
import sys
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from ai_experiments import merchants as M
from ai_experiments.evals.tracker import Run
from ai_experiments.paths import ROOT
from ai_experiments.retrieval import MerchantRetriever
from ai_experiments.scoring import Scorer, aggregate, per_item_path, write_records

MODELS = sys.argv[1:] or ["Qwen/Qwen2.5-0.5B", "Qwen/Qwen2.5-3B"]
SMOKE = bool(os.environ.get("SMOKE"))
SEED = 0
OUT = ROOT / "results" / f"retrieval_merchants{'_smoke' if SMOKE else ''}.json"

all_m = M.build()
rng = random.Random(SEED)
held_names = {m["name"] for c in M.CATEGORY_LIST for m in rng.sample([x for x in all_m if x["category"] == c], 2)}
train_m = [m for m in all_m if m["name"] not in held_names]
by_name = {m["name"]: m for m in all_m}
items = [it for it in M.eval_items(all_m) if it["task"] in ("bank_category", "clean_category")]
if SMOKE:
    items = [it for t in ("bank_category", "clean_category") for it in [x for x in items if x["task"] == t][:24]]
for it in items:  # the scorer wants id / level; the level carries the split so aggregate() reports held-out apart
    it["id"] = f"{it['task']}:{it['merchant']}"
    it["level"] = f"{it['task']}_{'heldout' if it['merchant'] in held_names else 'train'}"

results = {}


def query_of(it):
    return M.bank_string(by_name[it["merchant"]]) if it["task"] == "bank_category" else it["merchant"]


def with_records(it, records):
    """Prepend records right before the final query line, as section 4's oracle context did."""
    head, _, tail = it["prompt"].rpartition("\n\n")
    return dict(it, prompt=f"{head}\n\n" + "".join(f"Record: {r}\n" for r in records) + tail)


with Run("retrieval_merchants", model=",".join(MODELS), config=dict(seed=SEED, n_merchants=len(all_m), n_heldout=len(held_names), n_items=len(items), smoke=SMOKE), enabled=not SMOKE) as run:
    # ---------------------------------------------------------------- retrievers
    for tag, train in (("zero_shot", False), ("category_tuned", "category"), ("record_tuned", "record")):
        R = MerchantRetriever(all_m, train_m, train=train)
        r = {}
        for split, ms in (("train", train_m), ("heldout", [m for m in all_m if m["name"] in held_names])):
            for fmt, q in (("bank", [M.bank_string(m) for m in ms]), ("name", [m["name"] for m in ms])):
                rec = R.recall(q, [m["name"] for m in ms])
                r.update({f"{fmt}_{split}_{k}": v for k, v in rec.items()})
        results[f"retriever_{tag}"] = r; run.log(r, condition=f"retriever_{tag}")
        print(f"   retriever_{tag}: {r}", flush=True)
        if train == "record":
            RET = R
    top = RET.topk([query_of(it) for it in items], 3)
    ctx = {"ret1": [[RET.docs[t[0]]] for t in top], "ret3": [[RET.docs[i] for i in t] for t in top],
           "oracle": [[M.raw_fact(by_name[it["merchant"]])] for it in items]}
    results["retrieval_hits"] = {"ret1_gold": round(100 * sum(it["merchant"] in RET.keys[t[0]] for it, t in zip(items, top)) / len(items), 1)}
    run.log(results["retrieval_hits"], condition="retrieval_hits")
    del R, RET; torch.cuda.empty_cache()
    # ---------------------------------------------------------------- LLMs
    for model_name in MODELS:
        tok = AutoTokenizer.from_pretrained(model_name); tok.padding_side = "right"
        model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.bfloat16).cuda().eval()
        sc = Scorer(model, tok, maxlen=768, extras=False)
        short = model_name.split("/")[-1]
        for cond in ("none", "oracle", "ret1", "ret3"):
            t0 = time.time()
            its = items if cond == "none" else [with_records(it, c) for it, c in zip(items, ctx[cond])]
            recs = sc.score(its, label=f"{short} {cond}")
            r = aggregate(recs); r["eval_minutes"] = round((time.time() - t0) / 60, 2)
            results[f"{short}_{cond}"] = r; run.log(r, condition=f"{short}_{cond}")
            print(f"   {short} {cond}: {r}", flush=True)
            if not SMOKE:
                run.artifact(write_records(per_item_path(OUT, f"{short}_{cond}"), recs))
        del model, sc; torch.cuda.empty_cache()
    OUT.parent.mkdir(exist_ok=True); OUT.write_text(json.dumps(results, indent=2))
    if not SMOKE:
        run.artifact(OUT)
print(f"\nwrote {OUT.relative_to(ROOT)}")
