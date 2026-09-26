"""Curriculum ladder v2: does mixing symbol-tuning episodes into knowledge injection produce a
model that does the "Timmy" label-induction task from its own weights, without eroding ICL?

usage: uv run python scripts/exp_curriculum.py ARM [model] [steps] [lr]

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
(SST-2/Banking77/DBpedia/Subj, symbol + natural labels), general-text perplexity. The items are
the frozen sets in data/processed/ (ai_experiments.items; their hashes go into the tracker config),
and every eval also writes per-item, per-option log-probs to results/per_item/ (ai_experiments.scoring).
Always uses unsloth FastLanguageModel + LoRA r64 (like exp_universe_ladder.py ... unsloth).

EVAL_ONLY=1 re-scores the saved adapter of a trained arm instead of training (base arms only ever score).
PERIODIC=N evaluates a fixed subsample (every 4th ladder item, every 2nd ICL suite item, the 200
ARC-Easy known-facts items as the forgetting proxy) before training and every N steps, logged to the
tracker under condition "periodic" with the step; results and adapter get a "_pN" suffix so the
original arm stays (PLAN step 11). The final evaluation also scores the known-facts set (K_arc_easy)
whenever data/processed/known_facts_v1.json exists.
"""
import json
import os
import random
import sys
import time
from collections import defaultdict
from ai_experiments.paths import ROOT

import unsloth  # noqa: F401  (before transformers)
import torch
from unsloth import FastLanguageModel

from ai_experiments import universe as U
from ai_experiments import icl_suite as S
from ai_experiments import items as I
from ai_experiments.evals.tracker import Run
from ai_experiments.merchants import GENERAL_TEXT
from ai_experiments.scoring import Scorer, aggregate, per_item_path, perplexity, write_records

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
OUT = ROOT / "results" / f"curriculum_{tag}_{ARM}.json"
ADAPTER = ROOT / "models" / "adapters" / f"curriculum_{tag}_{ARM}_lora"
torch.manual_seed(SEED)

species = U.build(morph_p=MORPH_P)
K_texts = U.training_texts(species)
FROZEN = I.load_all(morph=MORPH_P > 0)  # never regenerated: the same items for every arm and commit
ladder, probes, suite = FROZEN.ladder, FROZEN.probes, FROZEN.suite
SMOKE = bool(os.environ.get("SMOKE"))
if SMOKE:
    # Quick end-to-end plumbing check, not an experiment: 2 optimizer steps, 1/40 of the eval items,
    # results in results/*_smoke.json, adapter under models/smoke/ (outside the DVC-tracked
    # models/adapters/), and nothing recorded in the tracker.
    STEPS = min(STEPS, 2)
    ladder, probes, suite = ladder[::40], probes[::12], suite[::48]
    OUT = OUT.with_name(OUT.stem + "_smoke.json")
    ADAPTER = ROOT / "models" / "smoke" / ADAPTER.name
PERIODIC = int(os.environ.get("PERIODIC", "0"))  # mid-training evaluation every N steps; 0 = off
known = FROZEN.known
if PERIODIC:
    OUT = OUT.with_name(OUT.stem + f"_p{PERIODIC}.json")
    ADAPTER = ADAPTER.with_name(ADAPTER.name.replace("_lora", f"_p{PERIODIC}_lora"))
print(f"arm {ARM} | {len(species)} species (morph_p={MORPH_P}) | {len(K_texts)} knowledge texts | items {FROZEN.version} "
      f"({'morph' if FROZEN.morph else 'plain'}): {len(ladder)} ladder | {len(probes)} probes | {len(suite)} ICL suite", flush=True)


def load():
    model, tok = FastLanguageModel.from_pretrained(MODEL, max_seq_length=MAXLEN, dtype=torch.bfloat16, load_in_4bit=False)
    tok.padding_side = "right"
    return tok, model


# ------------------------------------------------------------------ evaluation
def evaluate(model, tok):
    """Returns ({"noctx": metrics, "ctx": metrics}, {"noctx": records, "ctx": records}).
    Probes, ICL suite and perplexity live under noctx; records are per item (ai_experiments.scoring)."""
    model.eval(); torch.cuda.empty_cache(); t0 = time.time()
    sc = Scorer(model, tok, maxlen=MAXLEN)
    recs = {"noctx": sc.score(ladder, label="ladder") + sc.score(probes, label="probe") + sc.score(suite, label="ICL suite"),
            "ctx": sc.score(ladder, ctx=True, label="ladder+ctx")}
    if known:
        recs["noctx"] += sc.score(known, label="known facts")
    noctx = aggregate(recs["noctx"])
    sym = [v for k, v in noctx.items() if k.startswith("ICL_symbol")]
    nat = [v for k, v in noctx.items() if k.startswith("ICL_natural")]
    noctx["ICL_symbol_mean"] = round(sum(sym) / len(sym), 1)
    noctx["ICL_natural_mean"] = round(sum(nat) / len(nat), 1)
    noctx["L7_ppl_general"] = round(perplexity(model, tok, GENERAL_TEXT), 2)
    ctx = aggregate(recs["ctx"])
    noctx["eval_minutes"] = round((time.time() - t0) / 60, 1)
    return {"noctx": noctx, "ctx": ctx}, recs


def periodic_eval(model, tok):
    """Cheap mid-training point: a fixed subsample, no extra passes, no per-item file. Leaves the model in train mode."""
    model.eval(); torch.cuda.empty_cache(); t0 = time.time()
    sc = Scorer(model, tok, maxlen=MAXLEN, extras=False)
    m = aggregate(sc.score(ladder[::4]) + sc.score(suite[::2]) + (sc.score(known) if known else []))
    sym = [v for k, v in m.items() if k.startswith("ICL_symbol")]
    m["ICL_symbol_mean"] = round(sum(sym) / len(sym), 1)
    m["L7_ppl_general"] = round(perplexity(model, tok, GENERAL_TEXT), 2)
    m["eval_minutes"] = round((time.time() - t0) / 60, 1)
    model.train()
    return m


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
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    model.train()
    periodic = {}
    if PERIODIC:
        periodic[0] = periodic_eval(model, tok); run.log(periodic[0], condition="periodic", step=0)
        print(f"    step 0 periodic: {periodic[0]}", flush=True)
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
        if PERIODIC and (step + 1) % PERIODIC == 0 and step + 1 < STEPS:
            periodic[step + 1] = periodic_eval(model, tok); run.log(periodic[step + 1], condition="periodic", step=step + 1)
            print(f"    step {step + 1} periodic: {periodic[step + 1]}", flush=True)
    el = time.time() - t0
    return model, dict(train_minutes=round(el / 60, 1), train_tokens=tokens, tokens_per_s=round(tokens / el),
                       peak_alloc_GiB=round(torch.cuda.max_memory_allocated() / 2**30, 2),
                       **{f"n_{k}": v for k, v in counts.items()}), periodic


# ------------------------------------------------------------------ main
results = {}
phases = MIXTURES.get(ARM)
cfg = dict(arm=ARM, steps=STEPS if phases else 0, bs=BS, micro=MICRO, accum=ACCUM, lr=LR, seed=SEED, maxlen=MAXLEN,
           method="unsloth_lora", lora_r=64, lora_alpha=128, lora_targets="all_linear", morph_p=MORPH_P,
           mixture=json.dumps(phases), n_species=len(species), n_heldout=sum(s["heldout"] for s in species),
           n_knowledge_texts=len(K_texts), n_ladder_items=len(ladder), n_probes=len(probes), n_icl_items=len(suite),
           n_known_items=len(known), periodic=PERIODIC, **FROZEN.config())
with Run("curriculum_v2", model=MODEL, config=cfg, enabled=not SMOKE) as run:
    def save(recs, conds):
        """results JSON + tracker metrics, and one per-item JSONL per condition (conds maps noctx/ctx -> name)."""
        OUT.parent.mkdir(exist_ok=True); OUT.write_text(json.dumps(results, indent=2))
        for cond, mets in results.items():
            if cond != "periodic":  # already logged with steps during training
                run.log(mets, condition=cond)
        run.artifact(OUT)
        for key, cond in conds.items():
            run.artifact(write_records(per_item_path(OUT, cond), recs[key]))

    eval_only = bool(phases) and bool(os.environ.get("EVAL_ONLY")) and ADAPTER.exists()
    if eval_only:
        # re-score a previously trained adapter (e.g. after an eval-time crash) without retraining
        run.set_config(eval_only=True, adapter=str(ADAPTER.relative_to(ROOT)))
        model, tok = FastLanguageModel.from_pretrained(str(ADAPTER), max_seq_length=MAXLEN, dtype=torch.bfloat16, load_in_4bit=False)
        tok.padding_side = "right"
        r, recs = evaluate(model, tok)
        results["trained"], results["trained_ctx"] = r["noctx"], r["ctx"]
        if OUT.exists():  # keep the training-time stats recorded by the original run
            prev = json.loads(OUT.read_text()).get("trained", {})
            keep = ("train_minutes", "train_tokens", "tokens_per_s", "peak_alloc_GiB", "train_stats_note")
            results["trained"].update({k: v for k, v in prev.items() if k in keep or k.startswith("n_")})
        conds = {"noctx": "trained", "ctx": "trained_ctx"}
    elif not phases:
        tok, model = load()
        r, recs = evaluate(model, tok)
        results["base"], results["base_ctx"] = r["noctx"], r["ctx"]
        conds = {"noctx": "base", "ctx": "base_ctx"}
    else:
        tok, model = load()
        model, stats, periodic = train(model, tok, phases, run)
        model.save_pretrained(ADAPTER); print(f"   saved adapter -> {ADAPTER}", flush=True)
        r, recs = evaluate(model, tok)
        results["trained"] = {**r["noctx"], **stats}; results["trained_ctx"] = r["ctx"]
        if periodic:
            results["periodic"] = {str(k): v for k, v in periodic.items()}  # curves; scripts/curves.py reads the tracker
        conds = {"noctx": "trained", "ctx": "trained_ctx"}
    save(recs, conds)

print(f"\n=== {tag} arm {ARM} ===")
for cond, mets in results.items():
    print(f"-- {cond}")
    for k, v in mets.items():
        print(f"   {k:32s} {v}")
