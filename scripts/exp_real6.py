"""Score the REAL-6 evaluation set (PLAN step 35; `ai_experiments.real6`, data/processed/real6_v1.json) per cell.

  llm       option log-probability (mean per token, the ladder's rule) of each user category name after a 24-shot prompt of the
            user's own history, on the base model or a saved adapter; `ctx` = the same with the merchant's fact-DB record before
            the query (the retrieval oracle). Per-item records to results/per_item/, so row 33 can pair its arms against these.
  encoder   prototype distance on a frozen encoder: the class vector is the mean embedding of the user's history strings
            (normalised) per category, from the same 24 shots the LLM sees (`shots`) or from the whole history (`full`); and
            the name-plus-centroid mix of section 36 (`mix`).
Every cell gets a 95% bootstrap interval over its items and the section 11 null band (gold permuted within the cell, predictions fixed).

usage: uv run python scripts/exp_real6.py llm [base|<adapter dir under models/adapters>]      MODEL=Qwen/Qwen2.5-3B
       uv run python scripts/exp_real6.py encoder [minilm|bge]
       SMOKE=1 scores every 10th item, no tracker
outputs: results/real6_<tag>.json, results/per_item/real6_<tag>.<cond>.jsonl; tracker experiment "real6"
"""
import json
import os
import random
import sys
import time
from collections import defaultdict

import numpy as np

from ai_experiments import merchants as M
from ai_experiments import real6 as R6
from ai_experiments.evals.tracker import Run
from ai_experiments.paths import ROOT

ROUTE = sys.argv[1] if len(sys.argv) > 1 else "llm"
WHAT = sys.argv[2] if len(sys.argv) > 2 else ("base" if ROUTE == "llm" else "minilm")
MODEL = os.environ.get("MODEL", "Qwen/Qwen2.5-3B")
SMOKE = bool(os.environ.get("SMOKE"))
ENCODERS = {"minilm": "sentence-transformers/all-MiniLM-L6-v2", "bge": "BAAI/bge-base-en-v1.5"}
CONDS = os.environ.get("CONDS", "noctx,ctx").split(",")  # which LLM conditions to score (a retrieval-trained adapter needs ctx only)
ENC_CTX = bool(os.environ.get("ENC_CTX"))  # encoder: the merchant's fact-DB record appended to the query string (row 33's retrieval condition)
if not R6.PATH.exists():
    R6.freeze()
DOC = R6.load()
ITEMS = DOC["items"][::10] if SMOKE else DOC["items"]
tag = (MODEL.split("/")[-1] if WHAT == "base" else WHAT) if ROUTE == "llm" else WHAT
OUT = ROOT / "results" / f"real6_{tag}{'_smoke' if SMOKE else ''}.json"
CELLS = sorted({it["level"] for it in DOC["items"]})


def summarize(recs, rng=random.Random(0), n_boot=1000):
    """Per cell: accuracy, 95% bootstrap interval, null band; plus aggregates over seen/unseen and name types."""
    by = defaultdict(list)
    for r in recs:
        by[r["level"]].append(r)
    for r in recs:  # aggregates
        seen, nt = r["level"].split("_")[1], r["level"].split("_")[2]
        by[f"R6_{seen}_all"].append(r); by[f"R6_all_{nt}"].append(r); by["R6_all"].append(r)
    out = {}
    for lv, rs in sorted(by.items()):
        c = np.array([r["correct"] for r in rs], float); n = len(c)
        acc = 100 * c.mean()
        g1, g2 = np.random.default_rng(1), np.random.default_rng(2)
        boot = sorted(100 * c[g1.integers(0, n, n)].mean() for _ in range(n_boot)) if n > 1 else [acc, acc]
        preds = np.array([r["pred"] for r in rs]); golds = np.array([r["answer"] for r in rs])
        null = sorted(100 * (preds == g2.permutation(golds)).mean() for _ in range(n_boot)) if n > 1 else [acc, acc]
        out[lv] = round(float(acc), 1); out[lv + "_n"] = n
        out[lv + "_ci"] = [round(float(boot[int(0.025 * n_boot)]), 1), round(float(boot[int(0.975 * n_boot) - 1]), 1)]
        out[lv + "_null"] = [round(float(null[int(0.025 * n_boot)]), 1), round(float(null[int(0.975 * n_boot) - 1]), 1)]
        out[lv + "_chance"] = round(float(np.mean([100 / len(r["options"]) for r in rs])), 1)
    return out


def write_recs(cond, recs):
    p = ROOT / "results" / "per_item" / f"real6_{tag}{'_smoke' if SMOKE else ''}.{cond}.jsonl"
    p.parent.mkdir(exist_ok=True)
    p.write_text("\n".join(json.dumps(r) for r in recs) + "\n")
    return p


def run_llm(run):
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    import unsloth  # noqa: F401
    import torch
    from unsloth import FastLanguageModel
    from ai_experiments.scoring import Scorer
    src = MODEL if WHAT == "base" else str(ROOT / "models" / "adapters" / WHAT)
    model, tok = FastLanguageModel.from_pretrained(src, max_seq_length=2048, dtype=torch.bfloat16)
    tok.padding_side = "right"; model.eval()
    sc = Scorer(model, tok, maxlen=2048, extras=False, rows_per_forward=16, tokens_per_forward=24576)
    results = {}
    for cond, ctx in (("noctx", False), ("ctx", True)):
        if cond not in CONDS:
            continue
        t0 = time.time()
        recs = sc.score(ITEMS, ctx=ctx, label=f"REAL-6 {cond}")
        for r, it in zip(recs, ITEMS):
            r.update(user=it["user"], merchant=it["merchant"], known=it["known"], options=it["options"])
        results[cond] = summarize(recs); results[cond]["minutes"] = round((time.time() - t0) / 60, 1)
        run.log({k: v for k, v in results[cond].items() if isinstance(v, (int, float))}, condition=cond)
        run.artifact(write_recs(cond, [{k: v for k, v in r.items() if k != "options"} for r in recs]))
    return results


def run_encoder(run):
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(ENC_SRC, device="cuda")
    enc = lambda texts: model.encode(texts, batch_size=256, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)  # noqa: E731
    results = {}
    users = {u["user"]: u for u in DOC["users"]}
    for cond in ("shots", "full", "mix"):
        t0 = time.time(); recs = []
        for uid, u in users.items():
            its = [it for it in ITEMS if it["user"] == uid]
            if not its:
                continue
            names = [c["name"] for c in u["categories"]]
            hist = u["history"] if cond != "shots" else [h for h in u["history"] if h["text"] in set(u["shots"])]
            E = enc([M.normalize(h["text"]) for h in hist]); labs = [h["label"] for h in hist]
            protos = np.stack([E[[i for i, l in enumerate(labs) if l == n]].mean(0) if n in labs else np.zeros(E.shape[1]) for n in names])
            if cond == "mix":
                protos = protos / (np.linalg.norm(protos, axis=1, keepdims=True) + 1e-8) + enc(names)
            protos = protos / (np.linalg.norm(protos, axis=1, keepdims=True) + 1e-8)
            Q = enc([M.normalize(it["text"]) + (f" {it['record']}" if ENC_CTX else "") for it in its])
            preds = (Q @ protos.T).argmax(1)
            for it, p in zip(its, preds):
                recs.append(dict(id=it["id"], level=it["level"], answer=it["answer"], pred=int(p), correct=bool(p == it["answer"]), user=uid, merchant=it["merchant"], known=it["known"], options=it["options"]))
        results[cond] = summarize(recs); results[cond]["minutes"] = round((time.time() - t0) / 60, 2)
        run.log({k: v for k, v in results[cond].items() if isinstance(v, (int, float))}, condition=cond)
        run.artifact(write_recs(cond, [{k: v for k, v in r.items() if k != "options"} for r in recs]))
    return results


ENC_SRC = ENCODERS.get(WHAT, str(ROOT / "models" / "adapters" / WHAT))  # a name from ENCODERS or a fine-tuned encoder directory under models/adapters
if ROUTE == "encoder" and ENC_CTX:
    tag += "_ctx"; OUT = ROOT / "results" / f"real6_{tag}{'_smoke' if SMOKE else ''}.json"
cfg = dict(route=ROUTE, what=WHAT, model=MODEL if ROUTE == "llm" else ENC_SRC, real6_version=DOC["version"], real6_sha=DOC["sha256"], n_items=len(ITEMS), shots=DOC["shots"], conds=CONDS, enc_ctx=ENC_CTX)
with Run("real6", model=cfg["model"], config=cfg, enabled=not SMOKE) as run:
    results = run_llm(run) if ROUTE == "llm" else run_encoder(run)
    OUT.parent.mkdir(exist_ok=True); OUT.write_text(json.dumps(results, indent=2)); run.artifact(OUT)
print(f"=== real6 {tag}")
for cond, r in results.items():
    print(f"-- {cond}: " + " ".join(f"{lv}={r[lv]}" for lv in CELLS + ["R6_seen_all", "R6_unseen_all", "R6_all"] if lv in r) + f" | chance {r.get('R6_all_chance')} null {r.get('R6_all_null')}")
