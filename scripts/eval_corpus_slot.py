"""PLAN step 154 (MODEL-24): readers on strands-decider's own evaluation sets (scripts/build_strands_corpus.sh), on the same rows its recipe
scores: the held-out short tasks (holdout_v5_norule: the "test" half by strands' partition_examples, then its seeded sample of 6,000,
as `strands-decider eval --limit 6000`), the multi-step evals (ContractNLI, MuSiQue, BoardgameQA, HotpotQA), the generated document
questions (v16, v18) and answer adequacy (HelpSteer2, generated). Accuracy = the top option is the gold, by task.
  FAMILY=slot     decider's one-slot readout (ai_experiments.corpus_slot) on MODEL (a Hub id: Qwen/Qwen3.5-2B-Base, Mapika/decider-2b,
                  ...), with ADAPTER = a LoRA from exp_corpus_slot.py; options in a fixed per-row shuffle, scores in order
  FAMILY=strands  strands' own evaluate_checkpoint on MODEL (a Hub id or checkpoint dir), at its fitted temperature
Writes results/corpus_eval_<family>_<reader>.json: {set: {task: {n, acc}}, ...} and the scoring minutes per set.
usage: FAMILY=slot MODEL=Mapika/decider-2b uv run --with "strands-decider==0.1.0" --with transformers==5.17.0 --with flash-linear-attention
       --with "peft>=0.21" --with torch==2.13.0 --with torchvision==0.28.0 python scripts/eval_corpus_slot.py
"""
import hashlib
import json
import os
import random
import sys
import time
from collections import defaultdict

from ai_experiments import corpus_slot as CS
from ai_experiments.paths import ROOT

FAMILY, MODEL = os.environ.get("FAMILY", "slot"), os.environ["MODEL"]
ADAPTER = os.environ.get("ADAPTER", "")
CORPUS = ROOT / os.environ.get("CORPUS", "data/external/strands_corpus")
BATCH = int(os.environ.get("BATCH", "8"))
SMOKE = int(os.environ.get("SMOKE", "0"))  # rows per set, for a smoke run
SETS = ["holdout_v5_norule", "multistep_v14_eval", "generated_v16_eval", "generated_v18_eval", "adequacy_hs2_eval", "adequacy_gen_eval"]
READER = ADAPTER or MODEL.split("/")[-1]


def rows_of(name):
    rows = [json.loads(l) for l in open(CORPUS / f"{name}.jsonl") if l.strip()]
    if name == "holdout_v5_norule":  # strands' partition_examples(..., "test", seed=0) then sample_examples(..., 6000, seed=0)
        rows = [r for i, r in enumerate(rows) if (hashlib.sha256(f"0:{i}".encode()).digest()[0] & 1) == 1]
        random.Random(0).shuffle(rows); rows = rows[:6000]
    return rows[:SMOKE] if SMOKE else rows


def slot_reader():
    import importlib
    import torch
    import torch.nn.functional as F
    from huggingface_hub import snapshot_download
    from transformers import AutoModelForCausalLM, AutoTokenizer
    sys.path.insert(0, snapshot_download("Mapika/decider-2b", allow_patterns=["decider/*"]))
    P = importlib.import_module("decider.prompt")
    tok = AutoTokenizer.from_pretrained(MODEL)
    lm = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).cuda().eval()
    if ADAPTER:
        from peft import PeftModel
        lm = PeftModel.from_pretrained(lm, str(ROOT / "models" / "adapters" / ADAPTER)).merge_and_unload().eval()

    @torch.no_grad()
    def predict(name, rows):
        out = []
        for k in range(0, len(rows), BATCH):
            chunk = rows[k:k + BATCH]
            built = [CS.render(P, tok, r, random.Random(f"{name}:{k + j}"), train=False) for j, r in enumerate(chunk)]
            T = -(-max(len(b["ids"]) for b in built) // 64) * 64
            ids = torch.full((len(built), T), tok.pad_token_id or 0, dtype=torch.long); att = torch.zeros_like(ids)
            for i, b in enumerate(built):
                ids[i, :len(b["ids"])] = torch.tensor(b["ids"]); att[i, :len(b["ids"])] = 1
            h = lm.model(input_ids=ids.cuda(), attention_mask=att.cuda()).last_hidden_state
            for i, b in enumerate(built):
                z = F.linear(h[i, b["slot"]], lm.lm_head.weight[torch.tensor(b["labs"], device="cuda")]).float()
                out.append(b["order"][int(z.argmax())])
        return out
    return predict


def strands_reader():
    from strands_decider.data.format import Example
    from strands_decider.evaluate import evaluate_checkpoint

    def predict(name, rows):
        rep = evaluate_checkpoint(MODEL, [Example.from_dict(r) for r in rows], batch_size=16)
        return rep  # the report itself (accuracy by task); predictions are not exposed
    return predict


if __name__ == "__main__":
    from ai_experiments.licences import open_licence
    open_licence(os.environ.get("LICENCE_OF", MODEL if "/" in MODEL else "Qwen/Qwen3.5-2B-Base"))
    predict = slot_reader() if FAMILY == "slot" else strands_reader()
    res = {}
    for name in SETS:
        rows = rows_of(name); t0 = time.time()
        if FAMILY == "slot":
            pred = predict(name, rows)
            by = defaultdict(list)
            for r, p in zip(rows, pred):
                by[r["task"]].append(p == r["label"]); by["_all"].append(p == r["label"])
            res[name] = {t: dict(n=len(v), acc=round(sum(v) / len(v), 4)) for t, v in by.items()}
        else:
            rep = predict(name, rows)
            res[name] = {t: dict(n=v["n"], acc=round(v["accuracy"], 4)) for t, v in rep["by_task"].items() if v}
            res[name]["_all"] = dict(n=rep["overall"]["n"], acc=round(rep["overall"]["accuracy"], 4))
        res[name]["_minutes"] = round((time.time() - t0) / 60, 2)
        print(f"{FAMILY} {READER} {name}: " + ", ".join(f"{t} {v['acc']:.3f} (n={v['n']})" for t, v in res[name].items() if t != "_minutes")
              + f" [{res[name]['_minutes']} min]", flush=True)
    if not SMOKE:
        (ROOT / "results" / f"corpus_eval_{FAMILY}_{READER}.json").write_text(json.dumps(dict(family=FAMILY, model=MODEL, adapter=ADAPTER, sets=res), indent=2))
