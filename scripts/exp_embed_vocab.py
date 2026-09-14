"""Experiment: teach an embedding model (all-MiniLM-L6-v2) merchant -> category
knowledge and test whether adding merchant names as new vocabulary tokens helps.

Setup
  Train contrastively (in-batch negatives, MultipleNegativesRankingLoss-style):
  anchor = a sentence about the merchant (augmented set, mixed case, no category
  word), positive = the category's description text. Also anchor = bare name.
  Evaluate: nearest category text for (a) bare merchant name, (b) bank-statement
  string (uppercase, store number, city) that never appears in training.

Conditions
  zero_shot          untouched model
  ft_subword         fine-tune, original tokenizer
  ft_newtok_random   fine-tune, 120 merchant tokens added with random init
  ft_newtok_mean     fine-tune, 120 merchant tokens added with mean-of-subword init
Held-out generalization probe: 24 merchants (2/category) whose descriptions are
NEVER trained; their bank strings test whether the model learned "merchant ->
category" only by rote or also picks up on product words in context.
Runs on transformers + torch only.
"""
import json
import random
import sys
import time
from pathlib import Path
from ai_experiments.paths import ROOT

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

from ai_experiments import merchants as M
from ai_experiments.evals.tracker import Run

MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EPOCHS, BS, LR, SEED = 6, 32, 3e-5, 0
OUT = ROOT / "results" / "embed_vocab.json"
torch.manual_seed(SEED)

all_m = M.build()
rng = random.Random(SEED)
held = {m["name"] for c in M.CATEGORY_LIST for m in rng.sample([x for x in all_m if x["category"] == c], 2)}
train_m = [m for m in all_m if m["name"] not in held]
held_m = [m for m in all_m if m["name"] in held]
CAT_TEXT = {c: f"{c}: {', '.join(M.CATEGORIES[c])}" for c in M.CATEGORY_LIST}
print(f"{len(train_m)} train merchants, {len(held_m)} held-out merchants")


def load():
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModel.from_pretrained(MODEL).cuda()
    return tok, model


def embed(model, tok, texts, grad=False):
    b = tok(texts, padding=True, truncation=True, max_length=96, return_tensors="pt").to("cuda")
    ctx = torch.enable_grad() if grad else torch.no_grad()
    with ctx, torch.autocast("cuda", dtype=torch.bfloat16):
        out = model(**b).last_hidden_state
        mask = b["attention_mask"].unsqueeze(-1).to(out.dtype)
        return F.normalize((out * mask).sum(1) / mask.sum(1), dim=-1).float()


def evaluate(model, tok):
    model.eval()
    cats = embed(model, tok, [CAT_TEXT[c] for c in M.CATEGORY_LIST])
    def acc(texts, labels):
        e = embed(model, tok, texts)
        pred = (e @ cats.T).argmax(1).cpu()
        return round(100 * (pred == torch.tensor(labels)).float().mean().item(), 1)
    r = {}
    for tag, ms in (("train", train_m), ("heldout", held_m)):
        labels = [M.CATEGORY_LIST.index(m["category"]) for m in ms]
        r[f"name_{tag}"] = acc([m["name"] for m in ms], labels)
        r[f"bank_{tag}"] = acc([M.bank_string(m) for m in ms], labels)
        r[f"desc_{tag}"] = acc([M.raw_fact(m) for m in ms], labels)
    return r


def train(model, tok):
    pairs = []
    for m in train_m:
        pos = CAT_TEXT[m["category"]]
        pairs += [(t, pos) for t in M.augmented(m)]
        pairs += [(m["name"], pos)] * 3
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    model.train()
    rng = random.Random(SEED)
    t0 = time.time()
    for ep in range(EPOCHS):
        rng.shuffle(pairs)
        for i in range(0, len(pairs), BS):
            chunk = pairs[i:i + BS]
            a = embed(model, tok, [p[0] for p in chunk], grad=True)
            p = embed(model, tok, [p[1] for p in chunk], grad=True)
            scores = a @ p.T * 20.0
            # positives sharing a category are not negatives: mask duplicates
            same = torch.tensor([[x[1] == y[1] for y in chunk] for x in chunk], device="cuda")
            scores = scores.masked_fill(same & ~torch.eye(len(chunk), dtype=torch.bool, device="cuda"), -1e4)
            loss = F.cross_entropy(scores, torch.arange(len(chunk), device="cuda"))
            loss.backward(); opt.step(); opt.zero_grad()
    print(f"    trained {EPOCHS} epochs x {len(pairs)} pairs in {time.time() - t0:.0f}s, final loss {loss.item():.3f}")


def add_tokens(model, tok, names, init):
    emb = model.get_input_embeddings().weight
    with torch.no_grad():
        means = [emb[tok(n, add_special_tokens=False)["input_ids"]].mean(0) for n in names]
        std = emb.std().item()
    tok.add_tokens([n.lower() for n in names])  # uncased WordPiece: match lowercase form
    model.resize_token_embeddings(len(tok), mean_resizing=False)
    emb = model.get_input_embeddings().weight
    with torch.no_grad():
        for n, v in zip(names, means):
            i = tok.convert_tokens_to_ids(n.lower())
            emb[i] = v if init == "mean" else torch.randn_like(v) * std


results = {}
def run(name, fn):
    print(f"\n== {name}", flush=True)
    results[name] = fn(); print("  ", results[name], flush=True)
    OUT.parent.mkdir(exist_ok=True); OUT.write_text(json.dumps(results, indent=2))
    run_.log(results[name], condition=name)

def zero_shot():
    tok, model = load(); return evaluate(model, tok)
def ft_subword():
    tok, model = load(); train(model, tok); return evaluate(model, tok)
def ft_newtok(init):
    def f():
        tok, model = load()
        names = [m["name"] for m in all_m]  # add tokens for held-out names too (they just stay untrained)
        before = sum(len(tok(n, add_special_tokens=False)["input_ids"]) for n in names) / len(names)
        add_tokens(model, tok, names, init)
        print(f"    added {len(names)} tokens ({init} init); avg subwords/name before: {before:.1f}")
        train(model, tok); return evaluate(model, tok)
    return f

run_ = Run("merchant_embed_vocab", model=MODEL, config=dict(epochs=EPOCHS, bs=BS, lr=LR, seed=SEED, n_merchants=len(all_m), n_heldout=len(held_m))).__enter__()
run("zero_shot", zero_shot)
run("ft_subword", ft_subword)
run("ft_newtok_random", ft_newtok("random"))
run("ft_newtok_mean", ft_newtok("mean"))

print("\n=== SUMMARY (12-way accuracy %, chance 8.3) ===")
cols = ["name_train", "bank_train", "desc_train", "name_heldout", "bank_heldout", "desc_heldout"]
print(f"{'condition':18s}" + "".join(f"{c:>14s}" for c in cols))
for k, r in results.items():
    print(f"{k:18s}" + "".join(f"{str(r[c]):>14s}" for c in cols))
