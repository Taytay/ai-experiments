"""Arm P: soft-label prompt (context) distillation on arm C's data (PLAN step 7, BASE-3).

usage: uv run python scripts/exp_distill.py [model] [steps] [lr] [T]

The with-context numbers are the ceiling everywhere in section 8 (recall 98.8 to 100, Timmy 76 to
82). Prompt distillation (Kujanpaa et al., 2412.14964; context distillation, Snell et al.) trains
the weights to reproduce what the model does when the field guide is in front of it: the teacher
is the same base model reading [field-guide entries of every species named] + text, the student is
the LoRA model reading the bare text, and the loss is KL(teacher || student) at temperature T over
the target span. Same three streams and mixture as arm C, same budget:

  K  knowledge texts (45%)   student: the text; teacher: entries of the species it names + the text;
                             KL over every text token. Type-lore sentences name no species and use
                             plain cross-entropy instead.
  E  episodes (40%)          generated with ctx_frac=0 (never with context); teacher gets the entries
                             of every species in the prompt; KL over the answer tokens
  R  generic replay (15%)    plain cross-entropy on the answer tokens, exactly as in arm C

Everything else (LoRA r=64 alpha=128 on all linear layers, lr 1e-4, 800 steps of 16, warmup 30,
linear decay, seed 0, max length 768) matches exp_curriculum.py, and the evaluation is the same
frozen ladder / probes / ICL suite with per-item records, so `curriculum_summary.py`,
`scorer_table.py` and `ci_table.py` pick arm P up as one more column. Tracker experiment
`curriculum_v2`, arm P, method prompt_distill_kl. Prediction from the survey: bare-format L1 recall
moves from about 12 toward the with-context 98.8, with better ICL retention than hard-label episodes.
SMOKE=1 does the usual 2-step plumbing check.
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
import torch.nn.functional as F
from unsloth import FastLanguageModel

from ai_experiments import icl_suite as S
from ai_experiments import items as I
from ai_experiments import universe as U
from ai_experiments.evals.tracker import Run
from ai_experiments.merchants import GENERAL_TEXT
from ai_experiments.scoring import Scorer, aggregate, per_item_path, perplexity, write_records

ARM = "P"
MODEL = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen2.5-3B"
STEPS = int(sys.argv[2]) if len(sys.argv) > 2 else 800
LR = float(sys.argv[3]) if len(sys.argv) > 3 else 1e-4
TEMP = float(sys.argv[4]) if len(sys.argv) > 4 else 2.0
MICRO, ACCUM, SEED, MAXLEN = 8, 2, 0, 768
BS = MICRO * ACCUM
MIX = dict(K=0.45, E=0.40, R=0.15)
tag = MODEL.split("/")[-1]
OUT = ROOT / "results" / f"curriculum_{tag}_{ARM}.json"
ADAPTER = ROOT / "models" / "adapters" / f"curriculum_{tag}_{ARM}_lora"
torch.manual_seed(SEED)

species = U.build()
by_name = {s["name"]: s for s in species}
names_sorted = sorted(by_name, key=len, reverse=True)
K_texts = U.training_texts(species)
FROZEN = I.load_all(morph=False)
ladder, probes, suite = FROZEN.ladder, FROZEN.probes, FROZEN.suite
SMOKE = bool(os.environ.get("SMOKE"))
if SMOKE:
    STEPS = min(STEPS, 2)
    ladder, probes, suite = ladder[::40], probes[::12], suite[::48]
    OUT = OUT.with_name(OUT.stem + "_smoke.json")
    ADAPTER = ROOT / "models" / "smoke" / ADAPTER.name
print(f"arm {ARM} (prompt distillation, T={TEMP}) | {len(species)} species | {len(K_texts)} knowledge texts | "
      f"items {FROZEN.version}: {len(ladder)} ladder | {len(probes)} probes | {len(suite)} ICL suite", flush=True)


def guide_for(text: str) -> str:
    """Field-guide entries of every species named in the text, in name order; '' if none."""
    found = [n for n in names_sorted if n in text]
    return "\n".join(U.entry(by_name[n]) for n in sorted(found))


# ------------------------------------------------------------------ evaluation (same as exp_curriculum.py)
def evaluate(model, tok):
    model.eval(); torch.cuda.empty_cache(); t0 = time.time()
    sc = Scorer(model, tok, maxlen=MAXLEN)
    recs = {"noctx": sc.score(ladder, label="ladder") + sc.score(probes, label="probe") + sc.score(suite, label="ICL suite"),
            "ctx": sc.score(ladder, ctx=True, label="ladder+ctx")}
    noctx = aggregate(recs["noctx"])
    sym = [v for k, v in noctx.items() if k.startswith("ICL_symbol")]
    nat = [v for k, v in noctx.items() if k.startswith("ICL_natural")]
    noctx["ICL_symbol_mean"] = round(sum(sym) / len(sym), 1)
    noctx["ICL_natural_mean"] = round(sum(nat) / len(nat), 1)
    noctx["L7_ppl_general"] = round(perplexity(model, tok, GENERAL_TEXT), 2)
    ctx = aggregate(recs["ctx"])
    noctx["eval_minutes"] = round((time.time() - t0) / 60, 1)
    return {"noctx": noctx, "ctx": ctx}, recs


# ------------------------------------------------------------------ data
class Stream:
    def __init__(self, items, rng):
        self.items, self.rng, self.order = items, rng, []

    def next(self):
        if not self.order:
            self.order = list(range(len(self.items))); self.rng.shuffle(self.order)
        return self.items[self.order.pop()]


def encode(tok, sample):
    """-> dict(ids, labels, t_ids, t_span) where labels mark the loss span in the student sequence
    (-100 elsewhere), t_ids is the teacher sequence (context + same tokens) or None for plain CE,
    and t_span the [start, end) of the same target tokens inside t_ids."""
    eos = [tok.eos_token_id]
    if isinstance(sample, str):  # knowledge text: full-sequence target
        ids = tok(sample, add_special_tokens=False)["input_ids"][: MAXLEN - 1] + eos
        guide = guide_for(sample)
        if not guide:
            return dict(ids=ids, labels=list(ids), t_ids=None, t_span=None)
        ctx = tok(guide + "\n\n", add_special_tokens=False)["input_ids"]
        return dict(ids=ids, labels=list(ids), t_ids=ctx + ids, t_span=(len(ctx), len(ctx) + len(ids)))
    ans = tok(sample["answer"], add_special_tokens=False)["input_ids"] + eos
    p = tok(sample["prompt"], add_special_tokens=False)["input_ids"][-(MAXLEN - len(ans)):]
    ids, labels = p + ans, [-100] * len(p) + ans
    if sample.get("src"):  # replay: hard labels
        return dict(ids=ids, labels=labels, t_ids=None, t_span=None)
    ctx = tok("Field guide:\n" + guide_for(sample["prompt"]) + "\n\n", add_special_tokens=False)["input_ids"]
    return dict(ids=ids, labels=labels, t_ids=ctx + ids, t_span=(len(ctx) + len(p), len(ctx) + len(ids)))


def pad_batch(seqs, pad):
    L = max(map(len, seqs))
    ids = torch.tensor([s + [pad] * (L - len(s)) for s in seqs], device="cuda")
    att = torch.tensor([[1] * len(s) + [0] * (L - len(s)) for s in seqs], device="cuda")
    return ids, att


# ------------------------------------------------------------------ training
def train(model, tok, run):
    model = FastLanguageModel.get_peft_model(
        model, r=64, lora_alpha=128, lora_dropout=0.0, bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        use_gradient_checkpointing="unsloth", random_state=SEED)
    params = [p for p in model.parameters() if p.requires_grad]
    print(f"   LoRA trainable: {sum(p.numel() for p in params) / 1e6:.0f}M params", flush=True)
    rng = random.Random(SEED)
    streams = {"K": Stream(K_texts, rng), "E": Stream(U.episodes(species, n=6000, seed=3, ctx_frac=0.0), rng),
               "R": Stream(S.replay_episodes(n=4000, seed=11), rng)}
    srcs, ws = zip(*MIX.items())
    opt = torch.optim.AdamW(params, lr=LR, weight_decay=0.0, betas=(0.9, 0.95))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 30) * max(0.0, 1 - s / STEPS))
    pad = tok.pad_token_id or 0
    counts, tokens, kl_tokens, t0 = defaultdict(int), 0, 0, time.time()
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    model.train()
    for step in range(STEPS):
        loss_acc, kl_acc, ce_acc = 0.0, 0.0, 0.0
        for _ in range(ACCUM):
            batch = []
            for _ in range(MICRO):
                src = rng.choices(srcs, ws)[0]; counts[src] += 1
                batch.append(encode(tok, streams[src].next()))
            ids, att = pad_batch([b["ids"] for b in batch], pad)
            tokens += int(att.sum())
            logits = model(input_ids=ids, attention_mask=att).logits  # student, bare inputs
            # hard-label CE on samples without a teacher (replay, lore sentences)
            ce = torch.zeros((), device="cuda")
            hard = [i for i, b in enumerate(batch) if b["t_ids"] is None]
            if hard:
                lab = torch.tensor([batch[i]["labels"] + [-100] * (ids.shape[1] - len(batch[i]["labels"])) for i in hard], device="cuda")
                ce = F.cross_entropy(logits[hard, :-1].float().reshape(-1, logits.shape[-1]), lab[:, 1:].reshape(-1), ignore_index=-100)
            # soft labels from the same base model reading the field guide (adapter disabled)
            kl = torch.zeros((), device="cuda")
            soft = [i for i, b in enumerate(batch) if b["t_ids"] is not None]
            if soft:
                t_ids, t_att = pad_batch([batch[i]["t_ids"] for i in soft], pad)
                with torch.no_grad(), model.disable_adapter():
                    t_logits = model(input_ids=t_ids, attention_mask=t_att).logits
                s_rows, t_rows = [], []
                for j, i in enumerate(soft):
                    a, b = batch[i]["t_span"]
                    n_ctx = a - (len(batch[i]["ids"]) - (b - a))  # teacher offset = context length
                    s_a, s_b = a - n_ctx, b - n_ctx  # same target tokens in the student sequence
                    s_rows.append(logits[i, s_a - 1:s_b - 1]); t_rows.append(t_logits[j, a - 1:b - 1])
                s_l = torch.cat(s_rows).float() / TEMP; t_l = torch.cat(t_rows).float() / TEMP
                kl = F.kl_div(F.log_softmax(s_l, -1), F.log_softmax(t_l, -1), log_target=True, reduction="batchmean") * TEMP ** 2
                kl_tokens += s_l.shape[0]
            loss = (kl + ce) / ACCUM
            loss.backward(); loss_acc += loss.item(); kl_acc += kl.item() / ACCUM; ce_acc += ce.item() / ACCUM
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
        if (step + 1) % 50 == 0 or step + 1 == STEPS:
            el = time.time() - t0
            print(f"    step {step + 1}/{STEPS} loss {loss_acc:.3f} (kl {kl_acc:.3f} ce {ce_acc:.3f}) {el:.0f}s {tokens / el:.0f} tok/s "
                  f"{torch.cuda.max_memory_allocated() / 2**30:.1f} GiB", flush=True)
            run.log(dict(train_loss=loss_acc, train_kl=kl_acc, train_ce=ce_acc), condition="train", step=step + 1)
    el = time.time() - t0
    return model, dict(train_minutes=round(el / 60, 1), train_tokens=tokens, kl_tokens=kl_tokens, tokens_per_s=round(tokens / el),
                       peak_alloc_GiB=round(torch.cuda.max_memory_allocated() / 2**30, 2), **{f"n_{k}": v for k, v in counts.items()})


# ------------------------------------------------------------------ main
results = {}
cfg = dict(arm=ARM, steps=STEPS, bs=BS, micro=MICRO, accum=ACCUM, lr=LR, seed=SEED, maxlen=MAXLEN, method="prompt_distill_kl",
           temperature=TEMP, teacher="same base model, field-guide entries of the named species in context, adapter disabled",
           lora_r=64, lora_alpha=128, lora_targets="all_linear", morph_p=0.0, mixture=json.dumps([MIX]), n_species=len(species),
           n_heldout=sum(s["heldout"] for s in species), n_knowledge_texts=len(K_texts), n_ladder_items=len(ladder),
           n_probes=len(probes), n_icl_items=len(suite), **FROZEN.config())
with Run("curriculum_v2", model=MODEL, config=cfg, enabled=not SMOKE) as run:
    eval_only = bool(os.environ.get("EVAL_ONLY")) and ADAPTER.exists()
    if eval_only:
        run.set_config(eval_only=True, adapter=str(ADAPTER.relative_to(ROOT)))
        model, tok = FastLanguageModel.from_pretrained(str(ADAPTER), max_seq_length=MAXLEN, dtype=torch.bfloat16, load_in_4bit=False)
        tok.padding_side = "right"
        r, recs = evaluate(model, tok)
        results["trained"], results["trained_ctx"] = r["noctx"], r["ctx"]
        if OUT.exists():
            prev = json.loads(OUT.read_text()).get("trained", {})
            keep = ("train_minutes", "train_tokens", "kl_tokens", "tokens_per_s", "peak_alloc_GiB")
            results["trained"].update({k: v for k, v in prev.items() if k in keep or k.startswith("n_")})
    else:
        model, tok = FastLanguageModel.from_pretrained(MODEL, max_seq_length=MAXLEN, dtype=torch.bfloat16, load_in_4bit=False)
        tok.padding_side = "right"
        model, stats = train(model, tok, run)
        model.save_pretrained(ADAPTER); print(f"   saved adapter -> {ADAPTER}", flush=True)
        r, recs = evaluate(model, tok)
        results["trained"] = {**r["noctx"], **stats}; results["trained_ctx"] = r["ctx"]
    OUT.parent.mkdir(exist_ok=True); OUT.write_text(json.dumps(results, indent=2))
    for cond, mets in results.items():
        run.log(mets, condition=cond)
    run.artifact(OUT)
    for key, cond in {"noctx": "trained", "ctx": "trained_ctx"}.items():
        run.artifact(write_records(per_item_path(OUT, cond), recs[key]))

print(f"\n=== {tag} arm {ARM} ===")
for cond, mets in results.items():
    print(f"-- {cond}")
    for k, v in mets.items():
        print(f"   {k:32s} {v}")
