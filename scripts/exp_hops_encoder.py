"""Hop limits for a trained encoder (PLAN row 71, MODEL-12): ModernBERT-large trained on fresh chains (build_hops.chains, the
'train' half of the name pools, so no test name is ever seen), tested on data/processed/hops_v1.json at k = 1 to 12.

  MODE=options  row 53's layout: [CLS] instruction [SEP] ([MASK] name) per option [SEP] statements and question [SEP]; each
                option's [MASK] read by Laya's head (2 transformer layers + MLP); cross-entropy over the six options. One pass.
      LOOP=R    the looped encoder (Saunshi et al. ICLR 2025): the middle block (layers BLOCK, default 9-17 of 28; three global
                attention layers among them) repeated r times with tied weights, r drawn from 1 to R per training step (r = 1 is the
                pretrained network), tested at every r in R_TEST. LOOP=0: the plain network.
  MODE=scratch  thinking slots, one hop per round: the input carries the path walked so far ("Path so far: a -> b") and a [MASK]
                before every statement plus one before "(the chain ends here)"; the head picks the statement that continues the
                path from its last name, or the end. Trained with the gold path at every position (teacher forcing); tested by
                writing the picked statement's right-hand name into the path and re-encoding, up to MAX_ROUNDS rounds; the answer is
                the name where it stops (wrong if not among the options). Rounds are the encoder's analogue of a chain of thought.
TRAIN_KS: the hop counts trained on ("1-3", default; "1-12" asks whether the plain network can learn long chains at all).
Writes results/per_item/hops_enc_<name>.<cond>.jsonl and results/hops_enc_<name>.json (accuracy by names and k per condition);
the trained model goes to models/adapters/hops_enc_<name>/.
usage: MODE=options LOOP=4 STEPS=3000 uv run python scripts/exp_hops_encoder.py
"""
import json
import os
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch import nn

from ai_experiments.evals.tracker import Run
from ai_experiments.paths import PROCESSED, ROOT

sys.path.insert(0, str(Path(__file__).parent))
import build_hops as BH  # noqa: E402

MODE = os.environ.get("MODE", "options"); assert MODE in ("options", "scratch")
LOOP = int(os.environ.get("LOOP", "0"))
BLOCK = [int(x) for x in os.environ.get("BLOCK", "9-17").split("-")]
R_TEST = [int(x) for x in os.environ.get("R_TEST", "1,2,3,4,6,8,12").split(",")] if LOOP else [1]
lo, hi = (int(x) for x in os.environ.get("TRAIN_KS", "1-3").split("-")); TRAIN_KS = list(range(lo, hi + 1))
STEPS = int(os.environ.get("STEPS", "3000")); BATCH = int(os.environ.get("BATCH", "32"))
LR, HEAD_LR = float(os.environ.get("LR", "3e-5")), float(os.environ.get("HEAD_LR", "1e-4"))
MAX_ROUNDS = int(os.environ.get("MAX_ROUNDS", "16"))
SEED = int(os.environ.get("SEED", "0"))
BASE = "answerdotai/ModernBERT-large"
NAME = f"{MODE}{'_loop' + str(LOOP) if LOOP else ''}_k{lo}-{hi}_st{STEPS}" + (f"_s{SEED}" if SEED else "")
INSTR = {"options": "Question: following the equalities from the start name, where does its chain end? Options:",
         "scratch": "Question: which line continues the path from its last name? Or does the chain end there?"}[MODE]
rng = random.Random(SEED); torch.manual_seed(SEED)


class Scorer(nn.Module):
    """Laya's DecisionModel head (as scripts/exp_encoder_mask.py) on ModernBERT, with an optional looped middle block."""

    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder; d = encoder.config.hidden_size
        layer = nn.TransformerEncoderLayer(d, max(1, d // 64), 4 * d, 0.1, batch_first=True, norm_first=True)
        self.head = nn.TransformerEncoder(layer, 2, enable_nested_tensor=False)
        self.scorer = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1))
        self.base_layers = list(encoder.layers)

    def set_loops(self, r):
        a, b = BLOCK; L = self.base_layers
        self.encoder.layers = nn.ModuleList(L[:a] + L[a:b + 1] * r + L[b + 1:])

    def forward(self, ids, att, pos, pmask):
        h = self.encoder(input_ids=ids, attention_mask=att).last_hidden_state
        h = self.head(h, src_key_padding_mask=~att.bool())
        m = torch.gather(h, 1, pos.clamp(min=0)[:, :, None].expand(-1, -1, h.size(-1)))
        return self.scorer(m).squeeze(-1).float().masked_fill(~pmask, -1e4)


def t(tok, s):
    return tok(s, add_special_tokens=False)["input_ids"]


def question(it):
    return it["question"] if "question" in it else f"Start at {it['chain'][0]}. Where does the chain from {it['chain'][0]} end?"


def encode_options(tok, it):
    ids = [tok.cls_token_id] + t(tok, INSTR) + [tok.sep_token_id]; pos = []
    for o in it["options"]:
        pos.append(len(ids)); ids += [tok.mask_token_id] + t(tok, " " + o)
    body = "".join(f"- {a} = {b}\n" for a, b in it["statements"]) + question(it)
    return ids + [tok.sep_token_id] + t(tok, body) + [tok.sep_token_id], pos


def encode_scratch(tok, it, path):
    ids = [tok.cls_token_id] + t(tok, INSTR) + [tok.sep_token_id] + t(tok, f"Start: {path[0]}. Path so far: " + " -> ".join(path)) + [tok.sep_token_id]
    pos = []
    for a, b in it["statements"]:
        pos.append(len(ids)); ids += [tok.mask_token_id] + t(tok, f" {a} = {b}\n")
    pos.append(len(ids)); ids += [tok.mask_token_id] + t(tok, " (the chain ends here)")
    return ids + [tok.sep_token_id], pos


def scratch_target(it, node):
    nxt = [j for j, (a, _) in enumerate(it["statements"]) if a == node]
    return nxt[0] if nxt else len(it["statements"])


def collate(tok, batch):
    L = -(-max(len(i) for i, _ in batch) // 128) * 128; K = max(len(p) for _, p in batch)
    ids = torch.full((len(batch), L), tok.pad_token_id); att = torch.zeros((len(batch), L), dtype=torch.long)
    pos = torch.full((len(batch), K), -1); pm = torch.zeros((len(batch), K), dtype=torch.bool)
    for b, (i, p) in enumerate(batch):
        ids[b, :len(i)] = torch.tensor(i); att[b, :len(i)] = 1; pos[b, :len(p)] = torch.tensor(p); pm[b, :len(p)] = True
    return ids.cuda(), att.cuda(), pos.cuda(), pm.cuda()


def episode(tok):
    it = BH.chains(rng, rng.choice(TRAIN_KS), rng.choice(["single", "multi"]), split="train")
    if MODE == "options":
        return encode_options(tok, it), it["answer"]
    step = rng.randint(0, it["k"]); path = it["chain"][:step + 1]
    return encode_scratch(tok, it, path), scratch_target(it, path[-1])


def train(tok, model):
    enc = [p for n, p in model.named_parameters() if n.startswith("encoder.")]; rest = [p for n, p in model.named_parameters() if not n.startswith("encoder.")]
    opt = torch.optim.AdamW([dict(params=enc, lr=LR), dict(params=rest, lr=HEAD_LR)], weight_decay=0.01)
    warm = max(1, STEPS // 20)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / warm) * max(0.0, 1 - s / STEPS))
    model.train(); t0 = time.time(); losses, accs = [], []
    for step in range(STEPS):
        if LOOP:
            model.set_loops(rng.randint(1, LOOP))
        eps = [episode(tok) for _ in range(BATCH)]
        ids, att, pos, pm = collate(tok, [e for e, _ in eps]); y = torch.tensor([y for _, y in eps]).cuda()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(ids, att, pos, pm)
        loss = nn.functional.cross_entropy(logits, y)
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
        losses.append(loss.item()); accs.append(float((logits.argmax(-1) == y).float().mean()))
        if (step + 1) % 100 == 0 or step + 1 == STEPS:
            print(f"    step {step + 1}/{STEPS} loss {np.mean(losses[-100:]):.3f} acc {np.mean(accs[-100:]):.3f} {time.time() - t0:.0f}s "
                  f"{torch.cuda.max_memory_allocated() / 2**30:.1f} GiB", flush=True)
    return dict(train_minutes=round((time.time() - t0) / 60, 1), final_loss=float(np.mean(losses[-200:])), final_train_acc=float(np.mean(accs[-200:])))


@torch.no_grad()
def logits_for(tok, model, batch):
    ids, att, pos, pm = collate(tok, batch)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        return model(ids, att, pos, pm).float()


@torch.no_grad()
def test(tok, model, items):
    model.eval(); conds = {}
    if MODE == "options":
        for r in R_TEST:
            if LOOP:
                model.set_loops(r)
            recs = []
            for b in range(0, len(items), BATCH):
                chunk = items[b:b + BATCH]
                lg = torch.log_softmax(logits_for(tok, model, [encode_options(tok, it) for it in chunk]), -1)
                for it, row in zip(chunk, lg):
                    lp = row[:len(it["options"])].tolist()
                    recs.append(dict(id=it["id"], k=it["k"], names=it["names"], answer=it["answer"], sum_lp=lp, pred=int(np.argmax(lp))))
            conds[f"r{r:02d}" if LOOP else "plain"] = recs
        return conds
    paths = {it["id"]: [it["chain"][0]] for it in items}; done = {}
    for rnd in range(MAX_ROUNDS + 1):
        live = [it for it in items if it["id"] not in done]
        for b in range(0, len(live), BATCH):
            chunk = live[b:b + BATCH]
            lg = logits_for(tok, model, [encode_scratch(tok, it, paths[it["id"]]) for it in chunk])
            for it, row in zip(chunk, lg):
                j = int(row[:len(it["statements"]) + 1].argmax())
                if j == len(it["statements"]) or rnd == MAX_ROUNDS:
                    done[it["id"]] = rnd
                else:
                    paths[it["id"]].append(it["statements"][j][1])
    recs = []
    for it in items:
        end = paths[it["id"]][-1]
        recs.append(dict(id=it["id"], k=it["k"], names=it["names"], answer=it["answer"], sum_lp=None,
                         pred=it["options"].index(end) if end in it["options"] else -1, rounds=done[it["id"]], path=paths[it["id"]],
                         path_ok=paths[it["id"]] == it["chain"]))
    return {"scratch": recs}


if __name__ == "__main__":
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(BASE)
    enc = AutoModel.from_pretrained(BASE, attn_implementation="sdpa"); enc.config.reference_compile = False
    model = Scorer(enc).cuda()
    print(NAME, "train ks", TRAIN_KS, "loop", LOOP, flush=True)
    info = train(tok, model) if STEPS else {}
    items = json.loads((PROCESSED / "hops_v1.json").read_text())["items"]
    t0 = time.time(); conds = test(tok, model, items); info["test_seconds"] = round(time.time() - t0)
    (ROOT / "results" / "per_item").mkdir(parents=True, exist_ok=True)
    acc = {}
    for cond, recs in conds.items():
        with open(ROOT / "results" / "per_item" / f"hops_enc_{NAME}.{cond}.jsonl", "w") as f:
            for r in recs:
                f.write(json.dumps(r) + "\n")
        by = defaultdict(list)
        for r in recs:
            by[(r["names"], r["k"])].append(r["pred"] == r["answer"])
        acc[cond] = {f"{n}_k{k:02d}": round(100 * float(np.mean(v)), 1) for (n, k), v in sorted(by.items())}
        print(cond, " ".join(f"{k}:{v:.0f}" for k, v in acc[cond].items()), flush=True)
    cfg = dict(mode=MODE, loop=LOOP, block=BLOCK, r_test=R_TEST, train_ks=TRAIN_KS, steps=STEPS, batch=BATCH, lr=LR, head_lr=HEAD_LR, seed=SEED, base=BASE)
    out = ROOT / "results" / f"hops_enc_{NAME}.json"
    out.write_text(json.dumps({"config": cfg, **info, "acc": acc}, indent=1))
    with Run("hops_encoder", model=BASE, config=cfg, enabled=STEPS >= 100) as run:
        for cond, a in acc.items():
            run.log({"acc_mean": sum(a.values()) / len(a), **a}, condition=cond)
        run.artifact(out)
    if STEPS:
        if LOOP:
            model.set_loops(1)
        out = ROOT / "models" / "adapters" / f"hops_enc_{NAME}"; out.mkdir(parents=True, exist_ok=True)
        torch.save({k: v.detach().clone().cpu() for k, v in model.state_dict().items()}, out / "model.pt")
        (out / "config.json").write_text(json.dumps(cfg, indent=1))
