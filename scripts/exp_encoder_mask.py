"""Encoder option scorer in Laya's layout (PLAN step 53, QUESTIONS.md MODEL-6): can a 400M bidirectional encoder with one scored [MASK]
per category do the categoriser's job, in one forward pass, at milliseconds per item?

Sequence (Laya's `build_sequence`, `references/laya_analysis.md` 2.1):
  [CLS] choice question: <instruction> [SEP] [MASK] <category 1> [MASK] <category 2> ... [SEP] <state> [SEP]
where the state is the query transaction, optionally the merchant's record (CTX=1), then the user's 24 shots as "statement -> category"
lines. The model is Laya's `DecisionModel` (Apache-2.0, convaiinnovations/laya; re-implemented below without the act head): ModernBERT-large,
a learned question-type vector, two fresh transformer layers, and an MLP scoring each [MASK]'s hidden state; softmax over the options.

Training: across users, the fold's users held out (FOLD=k: user id mod 4, as row 42), cross-entropy on the gold option, the options
shuffled per episode, the 24 shots drawn at random from the rest of the user's history (REAL-6: DB-only merchants excluded, as the SFT
categoriser). Scored on the held-out users' frozen items with their frozen shots and the item's option order.

env: INIT=mbert|laya (plain ModernBERT-large or the Laya checkpoint), STEPS (default 1500; 0 = score the initial model, zero-shot),
     BATCH=16, LR=3e-5 (encoder; the head 1e-4), CTX=0|1, FOLD=0..3, POI=<set> (train on a set's own users, poi1_v1), ITEMS_SET=<set>
     (score another frozen set; its users when it has them), MAXLEN=2048, SEED, SMOKE=1
out: results/per_item/real6_encmask_<init>_<sfx>[_<set>].<noctx|ctx>.jsonl (sum_lp = the log-softmax over options, so every table and
     ai_experiments.scorecard reads it), results/encmask_<sfx>.json, the model under models/adapters/encmask_<init>_<sfx>/
usage: FOLD=0 INIT=laya uv run python scripts/exp_encoder_mask.py
"""
import json
import math
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn

from ai_experiments import real6 as R6
from ai_experiments.evals.tracker import Run
from ai_experiments.paths import ROOT
from ai_experiments.real6_eval import summarize, write_recs

INIT = os.environ.get("INIT", "mbert"); assert INIT in ("mbert", "laya")
SMOKE = bool(os.environ.get("SMOKE"))
STEPS = 3 if SMOKE else int(os.environ.get("STEPS", "1500"))
BATCH, LR, HEAD_LR = int(os.environ.get("BATCH", "16")), float(os.environ.get("LR", "3e-5")), float(os.environ.get("HEAD_LR", "1e-4"))
CTX = bool(int(os.environ.get("CTX", "0")))
FOLD = int(os.environ.get("FOLD", "0"))
POI = os.environ.get("POI", "")
ITEMS_SET = os.environ.get("ITEMS_SET", "")
MAXLEN = int(os.environ.get("MAXLEN", "2048"))
SEED = int(os.environ.get("SEED", "0"))
RUN_TAG = os.environ.get("RUN_TAG", "")
BASE = "answerdotai/ModernBERT-large"
INSTR = "choice question: Which of this person's own budget categories would they file the transaction under? Their past filings follow the transaction."
assert not (POI and CTX), "POI-1's history rows carry no records"

TRAIN_DOC = json.loads((ROOT / "data" / "processed" / f"{POI}.json").read_text()) if POI else R6.load()
DBREC = TRAIN_DOC.get("fact_db", {})
DB_ONLY = set() if POI else R6.db_only_merchants()
TEST = TRAIN_DOC
if ITEMS_SET:
    _s = json.loads((ROOT / "data" / "processed" / f"{ITEMS_SET}.json").read_text())
    TEST = dict(items=_s["items"], users=_s.get("users", TRAIN_DOC["users"]))
SFX = f"{POI + '_' if POI else ''}{'st' + str(STEPS) if STEPS else 'zeroshot'}{'_' + RUN_TAG if RUN_TAG else ''}_f{FOLD}{'_ctx' if CTX else ''}"
NAME = f"encmask_{INIT}_{SFX}"
COND = "ctx" if CTX else "noctx"
rng = random.Random(SEED); torch.manual_seed(SEED)


class DecisionModel(nn.Module):
    """Laya's DecisionModel (convaiinnovations/laya rl_common.py, Apache-2.0) without the act/escalate head."""

    def __init__(self, encoder, head_layers=2, dropout=0.1):
        super().__init__()
        self.encoder = encoder; d = encoder.config.hidden_size
        layer = nn.TransformerEncoderLayer(d, max(1, d // 64), 4 * d, dropout, batch_first=True, norm_first=True)
        self.head = nn.TransformerEncoder(layer, head_layers, enable_nested_tensor=False)
        self.type_emb = nn.Embedding(3, d)
        self.scorer = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1))

    def forward(self, ids, att, pos, pmask):
        h = self.encoder(input_ids=ids, attention_mask=att).last_hidden_state + self.type_emb.weight[0]
        h = self.head(h, src_key_padding_mask=~att.bool())
        m = torch.gather(h, 1, pos.clamp(min=0)[:, :, None].expand(-1, -1, h.size(-1)))
        return self.scorer(m).squeeze(-1).float().masked_fill(~pmask, -1e4)


def load_model():
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(BASE)
    model = DecisionModel(AutoModel.from_pretrained(BASE, attn_implementation="sdpa"))
    if INIT == "laya":
        from huggingface_hub import hf_hub_download
        from safetensors.torch import load_file
        sd = load_file(hf_hub_download("convaiinnovations/laya", "model.safetensors"))
        sd = {k: v for k, v in sd.items() if not k.startswith("act_head") and k != "temperature"}
        missing, unexpected = model.load_state_dict(sd, strict=False)
        assert not unexpected and not [k for k in missing if not k.startswith("encoder.")] and len(missing) < 5, (missing[:5], unexpected[:5])
    return tok, model.cuda()


def line(r):
    return f"{r['text']} | ${r['amount']:.2f} | {r['weekday']}"


def encode(tok, names, query, shots, record=None):
    """input ids and the [MASK] positions, Laya's layout; the state is truncated from the end (the last shots) when too long."""
    head = tok(INSTR, add_special_tokens=False)["input_ids"]
    ids = [tok.cls_token_id] + head + [tok.sep_token_id]; pos = []
    for n in names:
        pos.append(len(ids)); ids += [tok.mask_token_id] + tok(" " + n, add_special_tokens=False)["input_ids"][:24]
    ids.append(tok.sep_token_id)
    state = f"Transaction: {line(query)}\n" + (f"Note: {record}\n" if record else "") + "Past transactions:\n" + "".join(f"{line(s)} -> {s['label']}\n" for s in shots)
    st = tok(state, add_special_tokens=False)["input_ids"][:max(0, MAXLEN - len(ids) - 1)]
    return ids + st + [tok.sep_token_id], pos


def collate(tok, batch):
    L = max(len(i) for i, _ in batch); K = max(len(p) for _, p in batch)
    ids = torch.full((len(batch), L), tok.pad_token_id); att = torch.zeros((len(batch), L), dtype=torch.long)
    pos = torch.full((len(batch), K), -1); pm = torch.zeros((len(batch), K), dtype=torch.bool)
    for b, (i, p) in enumerate(batch):
        ids[b, :len(i)] = torch.tensor(i); att[b, :len(i)] = 1; pos[b, :len(p)] = torch.tensor(p); pm[b, :len(p)] = True
    return ids.cuda(), att.cuda(), pos.cuda(), pm.cuda()


def train_episodes():
    users = [u for u in TRAIN_DOC["users"] if u["user"] % 4 != FOLD]
    while True:
        u = rng.choice(users)
        hist = [h for h in u["history"] if h["merchant"] not in DB_ONLY]
        q = rng.choice(hist); others = rng.sample([h for h in hist if h is not q], min(24, len(hist) - 1))
        names = [c["name"] for c in u["categories"]]; rng.shuffle(names)
        yield names, q, others, (DBREC.get(q["merchant"]) if CTX else None), names.index(q["label"])


def train(tok, model):
    enc = [p for n, p in model.named_parameters() if n.startswith("encoder.")]; rest = [p for n, p in model.named_parameters() if not n.startswith("encoder.")]
    opt = torch.optim.AdamW([dict(params=enc, lr=LR), dict(params=rest, lr=HEAD_LR)], weight_decay=0.01)
    warm = max(1, STEPS // 20)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / warm) * max(0.0, 1 - s / STEPS))
    gen = train_episodes(); model.train(); t0 = time.time(); losses = []
    for step in range(STEPS):
        eps = [next(gen) for _ in range(BATCH)]
        ids, att, pos, pm = collate(tok, [encode(tok, n, q, s, r) for n, q, s, r, _ in eps])
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(ids, att, pos, pm)
        loss = nn.functional.cross_entropy(logits, torch.tensor([y for *_, y in eps]).cuda())
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
        losses.append(loss.item())
        if (step + 1) % 50 == 0 or step + 1 == STEPS:
            print(f"    step {step + 1}/{STEPS} loss {np.mean(losses[-50:]):.3f} {time.time() - t0:.0f}s {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB", flush=True)
    return dict(train_minutes=round((time.time() - t0) / 60, 1), final_loss=float(np.mean(losses[-100:])) if losses else None)


@torch.no_grad()
def score(tok, model):
    model.eval()
    users = {u["user"]: u for u in TEST["users"]}
    items = [it for it in TEST["items"] if it["user"] % 4 == FOLD or ITEMS_SET]
    if os.environ.get("USERS", "") == "all":
        items = TEST["items"]
    if SMOKE:
        items = items[::20]
    recs, seqs = [], []
    for it in items:
        u = users[it["user"]]; by_text = {h["text"]: h for h in u["history"]}
        shots = [by_text[t] for t in u["shots"] if t in by_text]
        names = [o.strip() for o in it["options"]]
        seqs.append(encode(tok, names, dict(text=it["text"], amount=it["amount"], weekday=it["weekday"]), shots, it.get("record") if CTX else None))
    t0 = time.time(); out = []
    for k in range(0, len(seqs), 32):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out += [torch.log_softmax(x[x > -1e3], -1).tolist() for x in model(*collate(tok, seqs[k:k + 32]))]
    torch.cuda.synchronize(); ms_batched = 1000 * (time.time() - t0) / len(seqs)
    t0 = time.time()
    for s in seqs[:50]:  # latency one request at a time
        with torch.autocast("cuda", dtype=torch.bfloat16):
            model(*collate(tok, [s]))
    torch.cuda.synchronize(); ms_single = 1000 * (time.time() - t0) / min(50, len(seqs))
    for it, lp in zip(items, out):
        pred = int(np.argmax(lp))
        recs.append(dict(id=it["id"], level=it["level"], user=it["user"], answer=it["answer"], pred=pred, correct=pred == it["answer"], sum_lp=lp,
                         options=it["options"], n_tok=len(seqs[len(recs)][0])))
    return recs, dict(ms_per_item_batched=round(ms_batched, 2), ms_per_item_single=round(ms_single, 2), n_items=len(recs),
                      truncated_share=round(float(np.mean([len(i) >= MAXLEN for i, _ in seqs])), 3))


if __name__ == "__main__":
    cfg = dict(init=INIT, steps=STEPS, batch=BATCH, lr=LR, head_lr=HEAD_LR, ctx=CTX, fold=FOLD, poi=POI, items_set=ITEMS_SET, maxlen=MAXLEN, seed=SEED, base=BASE,
               train_sha=TRAIN_DOC["sha256"], model=NAME)
    with Run("encmask", model=BASE, config=cfg, enabled=not SMOKE) as run:
        tok, model = load_model()
        stats = train(tok, model) if STEPS else {}
        if STEPS and not SMOKE:
            from safetensors.torch import save_file
            d = ROOT / "models" / "adapters" / NAME; d.mkdir(parents=True, exist_ok=True)
            save_file({k: v.contiguous() for k, v in model.state_dict().items()}, d / "model.safetensors"); (d / "config.json").write_text(json.dumps(cfg, indent=1))
        recs, timing = score(tok, model)
        summ = summarize(recs); summ.update(timing)
        print(f"  {NAME} {COND}: all {summ['R6_all']} (n={summ['R6_all_n']}), {timing}", flush=True)
        run.log({k: v for k, v in summ.items() if isinstance(v, (int, float))}, condition=COND)
        run.artifact(write_recs(NAME + (f"_{ITEMS_SET}" if ITEMS_SET else ""), COND, recs, smoke=SMOKE))
        out = ROOT / "results" / f"{NAME}{'_' + ITEMS_SET if ITEMS_SET else ''}{'_smoke' if SMOKE else ''}.json"
        out.write_text(json.dumps(dict(config=cfg, train=stats, **{COND: summ}), indent=1)); run.artifact(out)
