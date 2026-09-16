"""Encoder-decoder arms (PLAN step 18, MODEL-2 / REAL-3): the curriculum on a span-prediction model, scored on the same frozen
items as the decoder arms, by option log-probability and by generation.

usage: uv run python scripts/exp_encoder.py ARM [model] [steps] [lr]
  ARM     base   no training, score the model
          F2     arm C's mixture as sequence-to-sequence: knowledge texts as span prediction (each attribute value masked with
                 probability 0.5 plus 15% random spans, T5 sentinels; WikiDYK 2505.12306), episodes and ICL replay as prompt -> answer
          F2A    knowledge texts only (arm A's stream) as span prediction
  model   google/flan-t5-large (default; WikiDYK's model), google/t5gemma-l-l-ul2 (the modern encoder-decoder follow-on)
  steps   800 (16 sequences per step, as the decoder arms); lr 1e-4 (full fine-tuning, bitsandbytes 8-bit AdamW, fp32 weights,
          bf16 autocast; 30-step warmup and linear decay as exp_curriculum)
  env     UNIVERSE_N=125|625 (the 1,000 / 5,000-species universes), RUN_TAG, SEED, SMOKE=1, EVAL_ONLY=1 (re-score the saved weights),
          LORA=64 (rank-64 adapter on all projections over a bf16 base instead of full fine-tuning; T5Gemma at 1e-4 full FT was damaged)

Scoring. Every ladder, held-out, probe, ICL-suite, known-facts (ARC-Easy) and reverse item is scored as the decoder-side log-probability
of each option given the prompt as encoder input (sum over the option's tokens, the same `sum_lp` record as ai_experiments.scoring,
so the tables and paired tests read it unchanged); the field-guide context conditions use the items' prompt_ctx. The recall and
reverse levels are also scored by greedy generation (exact match of the option text after stripping, `gen_*` keys), which is how
WikiDYK reports memorisation. The fact levels (L1, L2, L5, L6, L8) are scored a second time in the span-prediction format they were
trained in (the question with the sentinel where the answer goes; `*_span` conditions), since the plain prompt -> answer rendering
is a format the knowledge stream never showed the model. No perplexity: the encoder-decoder's likelihood of raw text is not comparable to the decoders'.
Results: results/encoder_<model>_<ARM><SFX>.json, per-item files under results/per_item/, weights (bf16) under models/adapters/.
"""
import json
import os
import random
import sys
import time

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import torch
import torch.nn.functional as F
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from ai_experiments import icl_suite as S
from ai_experiments import items as I
from ai_experiments import universe as U
from ai_experiments.evals.tracker import Run
from ai_experiments.paths import ROOT
from ai_experiments.scoring import aggregate, per_item_path, write_records

ARM = sys.argv[1] if len(sys.argv) > 1 else "F2"
MODEL = sys.argv[2] if len(sys.argv) > 2 else "google/flan-t5-large"
STEPS = int(sys.argv[3]) if len(sys.argv) > 3 else 800
LR = float(sys.argv[4]) if len(sys.argv) > 4 else 1e-4
BS, MAXLEN, MAX_TGT = 16, 512, 48
SEED = int(os.environ.get("SEED", "0"))
RUN_TAG = os.environ.get("RUN_TAG", "")
UNIVERSE_N = int(os.environ.get("UNIVERSE_N", "20"))
SMOKE = bool(os.environ.get("SMOKE"))
EVAL_ONLY = bool(os.environ.get("EVAL_ONLY"))
LORA = int(os.environ.get("LORA", "0"))  # LORA=r: a rank-r adapter (alpha 2r) on every attention and MLP projection instead of full fine-tuning
tag = MODEL.split("/")[-1]
SFX = (f"_s{SEED}" if SEED else "") + (f"_{RUN_TAG}" if RUN_TAG else "")
OUT = ROOT / "results" / f"encoder_{tag}_{ARM}{SFX}{'_smoke' if SMOKE else ''}.json"
WEIGHTS = ROOT / ("models/smoke" if SMOKE else "models/adapters") / f"encoder_{tag}_{ARM}{SFX}_{'lora' if LORA else 'full'}"
torch.manual_seed(SEED)
rng = random.Random(SEED)

MIXTURES = {"F2": [dict(K=0.45, E=0.40, R=0.15)], "F2A": [dict(K=1.0)]}
phases = MIXTURES.get(ARM)
species = U.build(n_per_type=UNIVERSE_N)
FROZEN = I.load_all(morph=False, n=UNIVERSE_N if UNIVERSE_N != 20 else None)
ladder, probes, suite, known, reverse = FROZEN.ladder, FROZEN.probes, FROZEN.suite, FROZEN.known, FROZEN.reverse
K_texts = U.training_texts(species)
if SMOKE:
    STEPS = min(STEPS, 2); ladder, probes, suite, known, reverse = ladder[::40], probes[::12], suite[::48], known[::40], reverse[::40]
GEN_LEVELS = ("L1_recall", "L1_recall_fmt", "L8_reverse_easy", "L8_reverse_hard")
ATTR_VALUES = sorted({str(s[a]) for s in species for a in ("type", "weakness", "habitat", "diet", "region")} | {f"stage-{k}" for k in (1, 2, 3)}, key=len, reverse=True)

tok = AutoTokenizer.from_pretrained(MODEL)
if LORA:
    from peft import LoraConfig, PeftModel, get_peft_model
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL, dtype=torch.bfloat16).cuda()  # bf16 base, adapter in fp32 (peft default)
    if EVAL_ONLY and WEIGHTS.exists():
        model = PeftModel.from_pretrained(model, str(WEIGHTS))
    else:
        targets = ["q", "k", "v", "o", "wi_0", "wi_1", "wo"] if "t5" in MODEL.lower() and "gemma" not in MODEL.lower() else ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        model = get_peft_model(model, LoraConfig(r=LORA, lora_alpha=2 * LORA, lora_dropout=0.0, target_modules=targets, task_type="SEQ_2_SEQ_LM"))
        model.print_trainable_parameters()
else:
    model = AutoModelForSeq2SeqLM.from_pretrained(str(WEIGHTS) if EVAL_ONLY and WEIGHTS.exists() else MODEL, dtype=torch.float32).cuda()
SENTINELS = [tok.convert_tokens_to_ids(f"<extra_id_{i}>") for i in range(100)]
HAS_SENTINELS = SENTINELS[0] is not None and SENTINELS[0] != tok.unk_token_id
# without T5 sentinels (T5Gemma's Gemma tokenizer), a masked span is the literal word "___" in the input and the target is the
# spans in order separated by " ; " (the same information, no special tokens)
def sentinel(k):
    return f"<extra_id_{k}>" if HAS_SENTINELS else "___"
print(f"arm {ARM} | {MODEL} | {sum(p.numel() for p in model.parameters()) / 1e6:.0f}M params | {len(species)} species | {len(K_texts)} knowledge texts | "
      f"{len(ladder)} ladder | {len(suite)} ICL suite | {len(known)} known | {len(reverse)} reverse | sentinels {HAS_SENTINELS}", flush=True)


# ------------------------------------------------------------------ training examples
def span_example(text):
    """T5 span prediction: every attribute value in the text is masked with probability 0.5, plus 15% of the remaining words
    in short random spans; the target lists the masked spans behind sentinels."""
    words = text.replace("\n", " \n ").split(" ")
    mask = [False] * len(words)
    for i, w in enumerate(words):
        core = w.strip(".,;:()'\"")
        core = core[:-5] if core.endswith("-type") else core
        if core in ATTR_VALUES or core.lower().replace("-", " ") in ("stage 1", "stage 2", "stage 3"):
            if rng.random() < 0.5:
                mask[i] = True
    n_extra = max(1, round(0.15 * len(words)))
    while n_extra > 0:
        i = rng.randrange(len(words)); L = rng.choice([1, 1, 2, 3])
        for j in range(i, min(len(words), i + L)):
            if not mask[j]:
                mask[j] = True; n_extra -= 1
    src, tgt, k, i = [], [], 0, 0
    while i < len(words):
        if mask[i]:
            j = i
            while j < len(words) and mask[j]:
                j += 1
            src.append(sentinel(k)); tgt.append((f"<extra_id_{k}> " if HAS_SENTINELS else "") + " ".join(words[i:j])); k += 1; i = j
        else:
            src.append(words[i]); i += 1
    if HAS_SENTINELS:
        tgt.append(f"<extra_id_{k}>")
    return " ".join(src).replace(" \n ", "\n"), (" ".join(tgt) if HAS_SENTINELS else " ; ".join(tgt))


def qa_example(sample):
    return sample["prompt"], sample["answer"].strip()


class Stream:
    def __init__(self, items, rng):
        self.items, self.rng, self.order = items, rng, []

    def next(self):
        if not self.order:
            self.order = list(range(len(self.items))); self.rng.shuffle(self.order)
        return self.items[self.order.pop()]


def encode_batch(pairs):
    enc = tok([p[0] for p in pairs], padding=True, truncation=True, max_length=MAXLEN, return_tensors="pt")
    lab = tok(text_target=[p[1] for p in pairs], padding=True, truncation=True, max_length=MAX_TGT, return_tensors="pt")["input_ids"]
    lab[lab == tok.pad_token_id] = -100
    return {k: v.cuda() for k, v in enc.items()}, lab.cuda()


def train():
    import bitsandbytes as bnb
    streams = {"K": Stream(K_texts, rng)}
    need = {k for ph in phases for k in ph}
    if "E" in need:
        streams["E"] = Stream(U.episodes(species, n=6000, seed=3), rng)
    if "R" in need:
        streams["R"] = Stream(S.replay_episodes(n=4000, seed=11), rng)
    model.gradient_checkpointing_enable()  # the Flan-T5-large smoke peaked at 21.8 GiB without it; T5Gemma-large would not fit
    params = [p for p in model.parameters() if p.requires_grad]
    opt = (torch.optim.AdamW if LORA else bnb.optim.AdamW8bit)(params, lr=LR, weight_decay=0.0, betas=(0.9, 0.95))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 30) * max(0.0, 1 - s / STEPS))
    model.train(); torch.cuda.reset_peak_memory_stats(); t0 = time.time(); counts, tokens = {}, 0
    for step in range(STEPS):
        mix = phases[min(len(phases) - 1, step * len(phases) // STEPS)]
        srcs, ws = zip(*mix.items())
        pairs = []
        for _ in range(BS):
            src = rng.choices(srcs, ws)[0]; counts[src] = counts.get(src, 0) + 1
            x = streams[src].next()
            pairs.append(span_example(x) if src == "K" else qa_example(x))
        enc, lab = encode_batch(pairs); tokens += int(enc["attention_mask"].sum()) + int((lab != -100).sum())
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = model(**enc, labels=lab).loss
        loss.backward(); torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
        if (step + 1) % 50 == 0:
            el = time.time() - t0
            print(f"    step {step + 1}/{STEPS} loss {loss.item():.3f} mix={mix} {el:.0f}s {tokens / el:.0f} tok/s {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB", flush=True)
    model.eval(); el = time.time() - t0
    return dict(train_minutes=round(el / 60, 1), train_tokens=tokens, tokens_per_s=round(tokens / el), peak_alloc_GiB=round(torch.cuda.max_memory_allocated() / 2**30, 2),
                **{f"n_{k}": v for k, v in counts.items()})


# ------------------------------------------------------------------ scoring
FACT_LEVEL_PREFIXES = ("L1_", "L2_", "L5_", "L6_", "L8_")  # the levels whose answers were trained as masked spans, not as prompt -> answer


def span_form(prompt, option):
    """The span-prediction rendering the knowledge stream was trained on: the question with a sentinel where the answer goes, the
    answer behind the same sentinel (T5), or the placeholder word and the bare answer (tokenizers without sentinels)."""
    return prompt + " " + sentinel(0), ((f"<extra_id_0> " if HAS_SENTINELS else "") + option.strip())


@torch.no_grad()
def score_items(items, ctx=False, label="", span=False):
    """One record per item: sum of decoder log-probs of each option given the prompt (encoder input). span=True renders prompt and
    options in the trained span-prediction format (the fact levels' own training format)."""
    t0 = time.time(); recs = []; model.eval()
    if span:
        reqs = [(span_form(it["prompt_ctx" if ctx else "prompt"], "")[0], [span_form("", o)[1] for o in it["options"]]) for it in items]
    else:
        reqs = [(it["prompt_ctx" if ctx else "prompt"], it["options"]) for it in items]
    flat = [(p, o) for p, opts in reqs for o in opts]
    sums, ntoks = [], []
    B = 64
    for i in range(0, len(flat), B):
        chunk = flat[i:i + B]
        enc = tok([c[0] for c in chunk], padding=True, truncation=True, max_length=MAXLEN, return_tensors="pt").to("cuda")
        lab = tok(text_target=[c[1].strip() for c in chunk], padding=True, truncation=True, max_length=MAX_TGT, return_tensors="pt")["input_ids"].cuda()
        valid = lab != tok.pad_token_id
        lab_in = lab.masked_fill(~valid, -100)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(**enc, labels=lab_in).logits.float()
        lp = torch.log_softmax(logits, dim=-1).gather(-1, lab.clamp(min=0).unsqueeze(-1)).squeeze(-1)
        lp = (lp * valid).sum(1); sums += lp.tolist(); ntoks += valid.sum(1).tolist()
    k = 0
    for it, (p, opts) in zip(items, reqs):
        s = sums[k:k + len(opts)]; n = ntoks[k:k + len(opts)]; k += len(opts)
        pred = max(range(len(opts)), key=lambda j: s[j])
        recs.append(dict(id=it["id"], level=it["level"], answer=it["answer"], n_prompt_tok=None, sum_lp=[round(x, 4) for x in s], n_tok=n,
                         n_bytes=[len(o.encode()) for o in opts], pred=pred, correct=pred == it["answer"]))
    print(f"    scored {len(items)} {label} items in {(time.time() - t0) / 60:.1f} min", flush=True)
    return recs


@torch.no_grad()
def generate_exact(items, ctx=False, span=False):
    """Greedy generation; exact match (case-insensitive, stripped) against the gold option text. -> {gen_<level>: acc}
    span=True prompts in the span-prediction format and strips the sentinel from the output."""
    hits = {}
    B = 32
    for i in range(0, len(items), B):
        chunk = items[i:i + B]
        prompts = [span_form(it["prompt_ctx" if ctx else "prompt"], "")[0] if span else it["prompt_ctx" if ctx else "prompt"] for it in chunk]
        enc = tok(prompts, padding=True, truncation=True, max_length=MAXLEN, return_tensors="pt").to("cuda")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            gen = model.generate(**enc, max_new_tokens=MAX_TGT, do_sample=False)
        for it, g in zip(chunk, tok.batch_decode(gen, skip_special_tokens=True)):
            gold = it["options"][it["answer"]].strip().lower().rstrip(".")
            out = g.split("<extra_id_1>")[0].strip().lower().rstrip(".")
            hits.setdefault(it["level"], []).append(out == gold or out.startswith(gold))
    return {f"gen_{lvl}": round(100 * sum(v) / len(v), 1) for lvl, v in hits.items()}


def evaluate():
    t0 = time.time()
    recs = {"noctx": score_items(ladder, label="ladder") + score_items(probes, label="probe") + score_items(suite, label="ICL suite") + (score_items(known, label="known") if known else []) + (score_items(reverse, label="reverse") if reverse else []),
            "ctx": score_items(ladder, ctx=True, label="ladder+ctx") + (score_items(reverse, ctx=True, label="reverse+ctx") if reverse else [])}
    noctx = aggregate(recs["noctx"]); ctxm = aggregate(recs["ctx"])
    sym = [v for k, v in noctx.items() if k.startswith("ICL_symbol")]; nat = [v for k, v in noctx.items() if k.startswith("ICL_natural")]
    if sym: noctx["ICL_symbol_mean"] = round(sum(sym) / len(sym), 1)
    if nat: noctx["ICL_natural_mean"] = round(sum(nat) / len(nat), 1)
    gen_items = [it for it in ladder + reverse if it["level"] in GEN_LEVELS]
    noctx.update(generate_exact(gen_items)); ctxm.update(generate_exact(gen_items, ctx=True))
    # the fact levels in the span-prediction format they were trained in (MODEL-2's "same objective for injection and extraction")
    fact_items = [it for it in ladder + reverse if it["level"].startswith(FACT_LEVEL_PREFIXES)]
    recs["span"] = score_items(fact_items, label="fact levels, span format", span=True)
    recs["span_ctx"] = score_items(fact_items, ctx=True, label="fact levels + ctx, span format", span=True)
    spanm, spanc = aggregate(recs["span"]), aggregate(recs["span_ctx"])
    spanm.update(generate_exact(gen_items, span=True)); spanc.update(generate_exact(gen_items, ctx=True, span=True))
    noctx["eval_minutes"] = round((time.time() - t0) / 60, 1)
    return {"noctx": noctx, "ctx": ctxm, "span": spanm, "span_ctx": spanc}, recs


# ------------------------------------------------------------------ main
cfg = dict(arm=ARM, steps=STEPS if phases else 0, bs=BS, lr=LR, seed=SEED, maxlen=MAXLEN, max_target=MAX_TGT, method=f"lora{LORA}_seq2seq" if LORA else "full_ft_adamw8bit_seq2seq", lora_r=LORA, run_tag=RUN_TAG,
           universe_n=UNIVERSE_N, mixture=json.dumps(phases), n_species=len(species), n_knowledge_texts=len(K_texts), n_ladder_items=len(ladder), eval_only=EVAL_ONLY, **FROZEN.config())
results = {}
with Run("encoder_v1", model=MODEL, config=cfg, enabled=not SMOKE) as run:
    if phases and not EVAL_ONLY:
        stats = train()
        WEIGHTS.parent.mkdir(parents=True, exist_ok=True)
        if LORA:
            model.save_pretrained(WEIGHTS); tok.save_pretrained(WEIGHTS)
        else:
            model.to(torch.bfloat16).save_pretrained(WEIGHTS); tok.save_pretrained(WEIGHTS); model.float()
        print(f"   saved weights -> {WEIGHTS}", flush=True)
    else:
        stats = {}
    r, recs = evaluate()
    key = "trained" if phases else "base"
    results[key] = {**r["noctx"], **stats}; results[key + "_ctx"] = r["ctx"]; results[key + "_span"] = r["span"]; results[key + "_span_ctx"] = r["span_ctx"]
    OUT.parent.mkdir(exist_ok=True); OUT.write_text(json.dumps(results, indent=2))
    for cond, mets in results.items():
        run.log(mets, condition=cond)
    run.artifact(OUT)
    for k, cond in (("noctx", key), ("ctx", key + "_ctx"), ("span", key + "_span"), ("span_ctx", key + "_span_ctx")):
        run.artifact(write_records(per_item_path(OUT, cond), recs[k]))
print(f"\n=== {tag} arm {ARM} ===")
for cond, mets in results.items():
    print(f"-- {cond}")
    for k, v in mets.items():
        print(f"   {k:32s} {v}")
