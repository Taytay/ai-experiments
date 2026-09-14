"""Curriculum ladder v2: does mixing symbol-tuning episodes into knowledge injection produce a
model that does the "Timmy" label-induction task from its own weights, without eroding ICL?

usage: uv run python experiments/exp_curriculum.py ARM [model] [steps] [lr]

ARM     data mixture (fractions of each batch)                 universe
base    no training, evaluate base model                       plain
base_m  no training, evaluate base model                       morphology (marker suffixes)
A       knowledge text 1.0                                     plain
B       episodes 1.0                                           plain
C       knowledge .45 + episodes .40 + generic replay .15      plain
Cn      knowledge .50 + episodes .50 (no replay)               plain
D       phase 1: knowledge .85 + replay .15;                   plain
        phase 2: episodes .85 + replay .15   (sequential)
E       same as C                                              morphology

Streams: knowledge = universe.training_texts (full-sequence LM loss); episodes = universe.episodes
(loss on the answer only; random labels, varied templates, weakness attribute held out, half with
field-guide context); replay = icl_suite.replay_episodes (AG News/Emotion/TREC/20NG, random or
natural labels, answer-only loss).

Eval (all arms, same items): the 7-level ladder without and with context, held-out-species
induction, morphology probes (marked vs plain never-seen names), the ICL regression suite
(SST-2/Banking77/DBpedia/Subj, symbol + natural labels), general-text perplexity.
Always uses unsloth FastLanguageModel + LoRA r64 (like exp_universe_ladder.py ... unsloth).
"""
import json
import math
import os
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import unsloth  # noqa: F401  (before transformers)
import torch
import torch.nn.functional as F
from unsloth import FastLanguageModel

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))
import universe as U  # noqa: E402
import icl_suite as S  # noqa: E402
from evals.tracker import Run  # noqa: E402
from merchants import GENERAL_TEXT  # noqa: E402

ARM = sys.argv[1] if len(sys.argv) > 1 else "base"
MODEL = sys.argv[2] if len(sys.argv) > 2 else "Qwen/Qwen2.5-3B"
STEPS = int(sys.argv[3]) if len(sys.argv) > 3 else 800
LR = float(sys.argv[4]) if len(sys.argv) > 4 else 1e-4
MICRO, ACCUM, SEED, MAXLEN = 8, 2, 0, 768
BS = MICRO * ACCUM
MIXTURES = {  # arm -> list of phases; each phase = dict(source -> fraction)
    "A": [dict(K=1.0)],
    "B": [dict(E=1.0)],
    "C": [dict(K=0.45, E=0.40, R=0.15)],
    "Cn": [dict(K=0.5, E=0.5)],
    "D": [dict(K=0.85, R=0.15), dict(E=0.85, R=0.15)],
    "E": [dict(K=0.45, E=0.40, R=0.15)],
}
MORPH_P = 0.7 if ARM in ("E", "base_m") else 0.0
tag = MODEL.split("/")[-1]
OUT = Path(__file__).parent.parent / "results" / f"curriculum_{tag}_{ARM}.json"
ADAPTER = Path(__file__).parent.parent / "outputs" / f"curriculum_{tag}_{ARM}_lora"
torch.manual_seed(SEED)

species = U.build(morph_p=MORPH_P)
by_name = {s["name"]: s for s in species}
ladder = U.ladder(species) + U.heldout_induction(species)
probes = U.probes(species)
suite = S.suite_items()
K_texts = U.training_texts(species)
if os.environ.get("SMOKE"):  # quick end-to-end check: subsample eval items
    ladder, probes, suite = ladder[::40], probes[::12], suite[::48]
    OUT = OUT.with_name(OUT.stem + "_smoke.json")
print(f"arm {ARM} | {len(species)} species (morph_p={MORPH_P}) | {len(K_texts)} knowledge texts | "
      f"{len(ladder)} ladder items | {len(probes)} probes | {len(suite)} ICL suite items", flush=True)


def load():
    model, tok = FastLanguageModel.from_pretrained(MODEL, max_seq_length=MAXLEN, dtype=torch.bfloat16, load_in_4bit=False)
    tok.padding_side = "right"
    return tok, model


# ------------------------------------------------------------------ evaluation
@torch.no_grad()
def option_scores(model, tok, prompt, options):
    p_ids = tok(prompt, add_special_tokens=False)["input_ids"][-(MAXLEN - 32):]
    seqs, spans = [], []
    for o in options:
        o_ids = tok(o, add_special_tokens=False)["input_ids"]
        seqs.append(p_ids + o_ids); spans.append((len(p_ids), len(p_ids) + len(o_ids)))
    L = max(map(len, seqs)); pad = tok.pad_token_id or 0
    ids = torch.tensor([s + [pad] * (L - len(s)) for s in seqs], device="cuda")
    att = torch.tensor([[1] * len(s) + [0] * (L - len(s)) for s in seqs], device="cuda")
    logits = model(input_ids=ids, attention_mask=att).logits.float()
    lp = F.log_softmax(logits[:, :-1], -1).gather(-1, ids[:, 1:].unsqueeze(-1)).squeeze(-1)
    return [lp[i, a - 1:b - 1].mean().item() for i, (a, b) in enumerate(spans)]


@torch.no_grad()
def perplexity(model, tok, text):
    ids = tok(text, return_tensors="pt")["input_ids"].cuda()
    return math.exp(model(input_ids=ids, labels=ids).loss.float().item())


def accuracy(model, tok, items, context=False):
    hit, n, margin = defaultdict(int), defaultdict(int), defaultdict(list)
    for it in items:
        if context:
            it = U.with_context(it, by_name)
        sc = option_scores(model, tok, it["prompt"], it["options"])
        pred = max(range(len(sc)), key=sc.__getitem__)
        hit[it["level"]] += pred == it["answer"]; n[it["level"]] += 1
        p = torch.softmax(torch.tensor(sc), 0).sort(descending=True).values
        margin[it["level"]].append((p[0] - p[1]).item())
    res = {lv: round(100 * hit[lv] / n[lv], 1) for lv in n}
    for lv in ("L6_unseen_recall", "L6_seen_recall_ctrl"):
        if lv in margin:
            res[lv + "_margin"] = round(sum(margin[lv]) / len(margin[lv]), 3)
    return res


def evaluate(model, tok):
    """Returns {"noctx": {...}, "ctx": {...}}. Probes, ICL suite and perplexity live under noctx."""
    model.eval(); t0 = time.time()
    noctx = accuracy(model, tok, ladder)
    noctx.update(accuracy(model, tok, probes))
    icl = accuracy(model, tok, suite)
    noctx.update(icl)
    sym = [v for k, v in icl.items() if k.startswith("ICL_symbol")]
    nat = [v for k, v in icl.items() if k.startswith("ICL_natural")]
    noctx["ICL_symbol_mean"] = round(sum(sym) / len(sym), 1)
    noctx["ICL_natural_mean"] = round(sum(nat) / len(nat), 1)
    noctx["L7_ppl_general"] = round(perplexity(model, tok, GENERAL_TEXT), 2)
    ctx = accuracy(model, tok, ladder, context=True)
    noctx["eval_minutes"] = round((time.time() - t0) / 60, 1)
    return {"noctx": noctx, "ctx": ctx}


# ------------------------------------------------------------------ training
class Stream:
    """Endless shuffled iterator over a list."""
    def __init__(self, items, rng):
        self.items, self.rng, self.order = items, rng, []

    def next(self):
        if not self.order:
            self.order = list(range(len(self.items))); self.rng.shuffle(self.order)
        return self.items[self.order.pop()]


def encode(tok, sample):
    """sample: str (full-sequence loss) or dict(prompt, answer) (answer-only loss). -> (ids, labels)"""
    eos = [tok.eos_token_id]
    if isinstance(sample, str):
        ids = tok(sample, add_special_tokens=False)["input_ids"][: MAXLEN - 1] + eos
        return ids, list(ids)
    a = tok(sample["answer"], add_special_tokens=False)["input_ids"] + eos
    p = tok(sample["prompt"], add_special_tokens=False)["input_ids"][-(MAXLEN - len(a)):]
    return p + a, [-100] * len(p) + a


def train(model, tok, phases, run):
    model = FastLanguageModel.get_peft_model(
        model, r=64, lora_alpha=128, lora_dropout=0.0, bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        use_gradient_checkpointing="unsloth", random_state=SEED)
    params = [p for p in model.parameters() if p.requires_grad]
    print(f"   LoRA trainable: {sum(p.numel() for p in params) / 1e6:.0f}M params", flush=True)
    rng = random.Random(SEED)
    streams = {"K": Stream(K_texts, rng)}
    need = {k for ph in phases for k in ph}
    if "E" in need:
        streams["E"] = Stream(U.episodes(species, n=6000, seed=3), rng)
    if "R" in need:
        streams["R"] = Stream(S.replay_episodes(n=4000, seed=11), rng)
    opt = torch.optim.AdamW(params, lr=LR, weight_decay=0.0, betas=(0.9, 0.95))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 30) * max(0.0, 1 - s / STEPS))
    pad = tok.pad_token_id or 0
    counts, tokens, t0 = defaultdict(int), 0, time.time()
    model.train()
    for step in range(STEPS):
        mix = phases[min(len(phases) - 1, step * len(phases) // STEPS)]
        srcs, ws = zip(*mix.items())
        loss_acc = 0.0
        for _ in range(ACCUM):
            batch = []
            for _ in range(MICRO):
                src = rng.choices(srcs, ws)[0]; counts[src] += 1
                batch.append(encode(tok, streams[src].next()))
            L = max(len(i) for i, _ in batch)
            ids = torch.tensor([i + [pad] * (L - len(i)) for i, _ in batch], device="cuda")
            lab = torch.tensor([l + [-100] * (L - len(l)) for _, l in batch], device="cuda")
            att = (torch.arange(L, device="cuda")[None] < torch.tensor([len(i) for i, _ in batch], device="cuda")[:, None]).long()
            tokens += int(att.sum())
            loss = model(input_ids=ids, attention_mask=att, labels=lab).loss / ACCUM
            loss.backward(); loss_acc += loss.item()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
        if (step + 1) % 50 == 0 or step + 1 == STEPS:
            el = time.time() - t0
            print(f"    step {step + 1}/{STEPS} loss {loss_acc:.3f} mix={mix} {el:.0f}s {tokens / el:.0f} tok/s "
                  f"{torch.cuda.max_memory_allocated() / 2**30:.1f} GiB", flush=True)
            run.log(dict(train_loss=loss_acc), condition="train", step=step + 1)
    el = time.time() - t0
    return model, dict(train_minutes=round(el / 60, 1), train_tokens=tokens, tokens_per_s=round(tokens / el),
                       peak_alloc_GiB=round(torch.cuda.max_memory_allocated() / 2**30, 2),
                       **{f"n_{k}": v for k, v in counts.items()})


# ------------------------------------------------------------------ main
results = {}
phases = MIXTURES.get(ARM)
cfg = dict(arm=ARM, steps=STEPS if phases else 0, bs=BS, micro=MICRO, accum=ACCUM, lr=LR, seed=SEED, maxlen=MAXLEN,
           method="unsloth_lora", lora_r=64, lora_alpha=128, lora_targets="all_linear", morph_p=MORPH_P,
           mixture=json.dumps(phases), n_species=len(species), n_heldout=sum(s["heldout"] for s in species),
           n_knowledge_texts=len(K_texts), n_ladder_items=len(ladder), n_probes=len(probes), n_icl_items=len(suite))
with Run("curriculum_v2", model=MODEL, config=cfg) as run:
    def save():
        OUT.parent.mkdir(exist_ok=True); OUT.write_text(json.dumps(results, indent=2))
        for cond, mets in results.items():
            run.log(mets, condition=cond)

    tok, model = load()
    if not phases:
        r = evaluate(model, tok)
        results["base"], results["base_ctx"] = r["noctx"], r["ctx"]
    else:
        model, stats = train(model, tok, phases, run)
        model.save_pretrained(ADAPTER); print(f"   saved adapter -> {ADAPTER}", flush=True)
        r = evaluate(model, tok)
        results["trained"] = {**r["noctx"], **stats}; results["trained_ctx"] = r["ctx"]
    save(); run.artifact(OUT)

print(f"\n=== {tag} arm {ARM} ===")
for cond, mets in results.items():
    print(f"-- {cond}")
    for k, v in mets.items():
        print(f"   {k:32s} {v}")
