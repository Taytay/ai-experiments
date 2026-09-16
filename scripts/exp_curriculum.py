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
Cg      knowledge .45 + episodes .35 + replay .15 + general .05  plain   (PLAN step 25, TRAIN-7)
A1      one knowledge text per species (universe.single_texts, all attributes, no paraphrases), LM loss
A1m     the same 136 texts under MASKED FINE-TUNING (Pan et al. 2510.09885, PLAN step 8, TRAIN-5): the
        sample is "Recover the original passage from the masked version.\nMasked: <text with a random
        5-95% of tokens replaced by <|fim_pad|>>\nOriginal: <text>", loss on the original only
C1      knowledge (one text per species) .45 + episodes .40 + replay .15
Cm      masked knowledge (one text per species) .45 + episodes .40 + replay .15
        Run these on Qwen/Qwen2.5-3B-Instruct (the paper's method presupposes an instruction-tuned
        model): uv run python scripts/exp_curriculum.py Cm Qwen/Qwen2.5-3B-Instruct
M0      arm C's mixture, but the fractions are LOSS WEIGHTS: the loss is the weighted sum of each
        stream's own mean token loss, so a stream's share of the gradient is its fraction whatever
        its token count (PLAN step 9, TRAIN-1). No other change: the ablation for M20/M40/M60.
M20/M40/M60  loss-weighted mixture with E = .20 / .40 / .60, R = .15, the rest split 2:1 between
        knowledge texts (K) and the self-teaching stream S (universe.self_teaching: completion,
        true/false and in-document multiple choice derived from the same facts, answer-only loss);
        episodes carry the loss on every INFERABLE demo label as well as the answer (all-answer
        loss after 2512.19879 B.1, restricted to labels whose group already appeared, since the
        labels are random strings): about 1.9x the label tokens of answer-only episodes.

Streams: knowledge = universe.training_texts (full-sequence LM loss); episodes = universe.episodes
(loss on the answer only; random labels, varied templates, weakness attribute held out, half with
field-guide context); replay = icl_suite.replay_episodes (AG News/Emotion/TREC/20NG, random or
natural labels, answer-only loss); general = icl_suite.general_replay_texts (WikiText-2 train paragraphs
cut to 70 words, full-sequence loss: pretraining-style replay, 5% of sequences but about a quarter of
the loss-bearing tokens, since knowledge texts are 24 tokens long). Per-stream sequence, token and
loss-bearing-token counts are recorded (n_K, tok_K, lb_K, ...).

Eval (all arms, same items): the 7-level ladder without and with context, held-out-species
induction, morphology probes (marked vs plain never-seen names), the ICL regression suite
(SST-2/Banking77/DBpedia/Subj, symbol + natural labels), general-text perplexity. The items are
the frozen sets in data/processed/ (ai_experiments.items; their hashes go into the tracker config),
and every eval also writes per-item, per-option log-probs to results/per_item/ (ai_experiments.scoring).
Always uses unsloth FastLanguageModel + LoRA r64 (like exp_universe_ladder.py ... unsloth).

EVAL_ONLY=1 re-scores the saved adapter of a trained arm instead of training (base arms only ever score).
SEED=N (default 0) seeds the LoRA init, the stream shuffles and the mixture draws; N > 0 adds "_sN" to the
results and adapter names so the seed-0 runs stay (PLAN step 10, STAT-1).
PERIODIC=N evaluates a fixed subsample (every 4th ladder item, every 2nd ICL suite item, the 200
ARC-Easy known-facts items as the forgetting proxy) before training and every N steps, logged to the
tracker under condition "periodic" with the step; results and adapter get a "_pN" suffix so the
original arm stays (PLAN step 11). The final evaluation also scores the known-facts set (K_arc_easy)
whenever data/processed/known_facts_v1.json exists, and every evaluation reports L7_ppl_wikitext, the
perplexity on the frozen WikiText-2 slice (data/processed/corpus_ppl_v1.json, PLAN step 24), beside the
one-paragraph L7_ppl_general.
"""
import json
import os
import random
import sys
import time
from collections import defaultdict
from ai_experiments.paths import ROOT

# The batched scorer's 64-row forwards fragment the caching allocator; without expandable segments the first
# backward after a periodic evaluation ran out of memory (REPORT.md 19). Must be set before CUDA initialises.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import unsloth  # noqa: F401  (before transformers)
import torch
from unsloth import FastLanguageModel

from ai_experiments import universe as U
from ai_experiments import icl_suite as S
from ai_experiments import items as I
from ai_experiments.evals.tracker import Run
from ai_experiments.merchants import GENERAL_TEXT
from ai_experiments.scoring import Scorer, aggregate, corpus_perplexity, per_item_path, perplexity, write_records

ARM = sys.argv[1] if len(sys.argv) > 1 else "base"
MODEL = sys.argv[2] if len(sys.argv) > 2 else "Qwen/Qwen2.5-3B"
STEPS = int(sys.argv[3]) if len(sys.argv) > 3 else 800
LR = float(sys.argv[4]) if len(sys.argv) > 4 else 1e-4
MICRO, ACCUM, MAXLEN = int(os.environ.get("MICRO", "16")), int(os.environ.get("ACCUM", "1")), 768  # MICRO x ACCUM = 16 sequences per step
GRAD_CKPT = os.environ.get("GRAD_CKPT", "1")  # "1" (default: plain torch checkpointing), "unsloth" (offloaded; three packed runs died
# of CUDA out-of-memory / cuBLAS errors raised in its backward within 200 steps, REPORT.md 20.4), or "0" (none: 30% faster in a 60-step bench but a full arm C run
# died of memory at step 600 after its periodic evaluations, REPORT.md 19)
GRAD_CKPT = {"0": False, "1": True}.get(GRAD_CKPT, GRAD_CKPT)
BENCH = int(os.environ.get("BENCH", "0"))  # BENCH=N: train N steps, print throughput, no eval / save / tracker
EXTRAS = bool(int(os.environ.get("EXTRAS", "0")))  # EXTRAS=1: also record dc/unc/mcf/hyb log-probs (the section 10 scorer study); doubles ladder time
PACK = int(os.environ.get("PACK", "2048"))  # PACK=T: pack each micro-batch's sequences into rows of at most T tokens (padding-free,
# block-diagonal attention through unsloth's packed_seq_lengths path; PLAN step 26, TRAIN-6). PACK=0 = one sequence per row (padded).
# Before 2026-09-15 every run used MICRO=8 ACCUM=2 PACK=0 (REPORT.md 19 measures the change: 2 to 2.5x faster, same numbers).
RUN_TAG = os.environ.get("RUN_TAG", "")  # optional suffix on the results and adapter names (validation runs, ablations)
SEED = int(os.environ.get("SEED", "0"))  # training seed: LoRA init, stream order, mixture draws (PLAN step 10, STAT-1)
BS = MICRO * ACCUM
MIXTURES = {  # arm -> list of phases; each phase = dict(source -> fraction)
    "A": [dict(K=1.0)],
    "B": [dict(E=1.0)],
    "C": [dict(K=0.45, E=0.40, R=0.15)],
    "Cn": [dict(K=0.5, E=0.5)],
    "D": [dict(K=0.85, R=0.15), dict(E=0.85, R=0.15)],
    "E": [dict(K=0.45, E=0.40, R=0.15)],
    "Cg": [dict(K=0.45, E=0.35, R=0.15, G=0.05)],
    "A1": [dict(K1=1.0)],
    "A1m": [dict(Km=1.0)],
    "C1": [dict(K1=0.45, E=0.40, R=0.15)],
    "Cm": [dict(Km=0.45, E=0.40, R=0.15)],
    "M0": [dict(K=0.45, E=0.40, R=0.15)],
    "M20": [dict(K=0.43, S=0.22, E=0.20, R=0.15)],
    "M40": [dict(K=0.30, S=0.15, E=0.40, R=0.15)],
    "M60": [dict(K=0.17, S=0.08, E=0.60, R=0.15)],
}
MASK_TOKEN = "<|fim_pad|>"  # a reserved single token of the Qwen2.5 vocabulary, never in any text
MASK_INSTRUCTION = "Recover the original passage from the masked version.\nMasked:"
MASK_RNG = random.Random(1000 + int(os.environ.get("SEED", "0")))
BY_LOSS = ARM.startswith("M")          # fractions are per-stream loss weights, not just sampling odds
ALL_ANSWER = BY_LOSS and ARM != "M0"   # episodes: loss on every demo label too
MORPH_P = 0.7 if ARM in ("E", "base_m") else 0.0
tag = MODEL.split("/")[-1]
SFX = (f"_s{SEED}" if SEED else "") + (f"_{RUN_TAG}" if RUN_TAG else "")  # seed 0, no tag keeps the original names
OUT = ROOT / "results" / f"curriculum_{tag}_{ARM}{SFX}.json"
ADAPTER = ROOT / "models" / "adapters" / f"curriculum_{tag}_{ARM}{SFX}_lora"
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
if BENCH:
    STEPS = BENCH
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
def wikitext_ppl(model, tok):
    """L7_ppl_wikitext (+ the standard error of its mean NLL) on the frozen corpus slice; {} if not frozen."""
    if not FROZEN.corpus:
        return {}
    cp = corpus_perplexity(model, tok, FROZEN.corpus, maxlen=MAXLEN)
    return {"L7_ppl_wikitext": cp["ppl"], "L7_nll_wikitext_se": cp["nll_se"]}


def evaluate(model, tok):
    """Returns ({"noctx": metrics, "ctx": metrics}, {"noctx": records, "ctx": records}).
    Probes, ICL suite and perplexity live under noctx; records are per item (ai_experiments.scoring)."""
    model.eval(); torch.cuda.empty_cache(); t0 = time.time()
    sc = Scorer(model, tok, maxlen=MAXLEN, extras=EXTRAS)
    recs = {"noctx": sc.score(ladder, label="ladder") + sc.score(probes, label="probe") + sc.score(suite, label="ICL suite"),
            "ctx": sc.score(ladder, ctx=True, label="ladder+ctx")}
    if known:
        recs["noctx"] += sc.score(known, label="known facts")
    if FROZEN.reverse:
        recs["noctx"] += sc.score(FROZEN.reverse, label="reverse")
        recs["ctx"] += sc.score(FROZEN.reverse, ctx=True, label="reverse+ctx")
    noctx = aggregate(recs["noctx"])
    sym = [v for k, v in noctx.items() if k.startswith("ICL_symbol")]
    nat = [v for k, v in noctx.items() if k.startswith("ICL_natural")]
    noctx["ICL_symbol_mean"] = round(sum(sym) / len(sym), 1)
    noctx["ICL_natural_mean"] = round(sum(nat) / len(nat), 1)
    noctx["L7_ppl_general"] = round(perplexity(model, tok, GENERAL_TEXT), 2)
    noctx.update(wikitext_ppl(model, tok))
    ctx = aggregate(recs["ctx"])
    noctx["eval_minutes"] = round((time.time() - t0) / 60, 1)
    torch.cuda.empty_cache()
    return {"noctx": noctx, "ctx": ctx}, recs


def periodic_eval(model, tok):
    """Cheap mid-training point: a fixed subsample, no extra passes, no per-item file. Leaves the model in train mode."""
    model.eval(); torch.cuda.empty_cache(); t0 = time.time()
    sc = Scorer(model, tok, maxlen=MAXLEN, extras=False, rows_per_forward=32, tokens_per_forward=16384)  # smaller footprint mid-training
    m = aggregate(sc.score(ladder[::4]) + sc.score(suite[::2]) + (sc.score(known) if known else []))
    sym = [v for k, v in m.items() if k.startswith("ICL_symbol")]
    m["ICL_symbol_mean"] = round(sum(sym) / len(sym), 1)
    m["L7_ppl_general"] = round(perplexity(model, tok, GENERAL_TEXT), 2)
    m.update(wikitext_ppl(model, tok))
    m["eval_minutes"] = round((time.time() - t0) / 60, 1)
    model.train(); torch.cuda.empty_cache()
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


def demo_label_spans(prompt, demos, window=12):
    """Character spans of the INFERABLE demo labels in an episode prompt: a demo's label counts only if the
    same label already appeared on an earlier demo (labels are random strings, so a group's first label
    is unpredictable and would only teach the label distribution). The occurrence used is the one that
    starts within `window` characters after the demo name ("X = l", "X 'l'", "X -> l", "Input: X\nOutput: l")."""
    import re
    spans, seen, hits = [], set(), []
    for name, label in demos:
        for m in re.finditer(re.escape(name), prompt):
            hit = re.compile(r"(?<![A-Za-z0-9])" + re.escape(label) + r"(?![A-Za-z0-9])").search(prompt, m.end(), m.end() + window + len(label))
            if hit and hit.start() - m.end() <= window:
                hits.append((hit.start(), hit.end(), label)); break
    for a, b, label in sorted(hits):
        if label in seen:
            spans.append((a, b))
        seen.add(label)
    return spans


def encode_masked(tok, text):
    """Masked fine-tuning sample: instruction + the text with a random 5-95% of its tokens replaced by the mask
    token, then "Original:" and the text itself, which alone carries the loss. Mask ratio drawn per sample."""
    eos = [tok.eos_token_id]
    ids = tok(" " + text, add_special_tokens=False)["input_ids"]
    mask_id = tok.convert_tokens_to_ids(MASK_TOKEN)
    t = MASK_RNG.uniform(0.05, 0.95)
    masked = [mask_id if MASK_RNG.random() < t else i for i in ids]
    p = tok(MASK_INSTRUCTION, add_special_tokens=False)["input_ids"] + masked + tok("\nOriginal:", add_special_tokens=False)["input_ids"]
    a = ids + eos
    return p + a, [-100] * len(p) + a


def encode(tok, sample):
    """sample: str (full-sequence loss), dict(mask=text) (masked fine-tuning) or dict(prompt, answer) (answer-only
    loss; with ALL_ANSWER and sample["demos"], the demo labels inside the prompt carry the loss too). -> (ids, labels)"""
    eos = [tok.eos_token_id]
    if isinstance(sample, dict) and "mask" in sample:
        return encode_masked(tok, sample["mask"])
    if isinstance(sample, str):
        ids = tok(sample, add_special_tokens=False)["input_ids"][: MAXLEN - 1] + eos
        return ids, list(ids)
    a = tok(sample["answer"], add_special_tokens=False)["input_ids"] + eos
    if ALL_ANSWER and sample.get("demos"):
        enc = tok(sample["prompt"], add_special_tokens=False, return_offsets_mapping=True)
        p, offs = enc["input_ids"], enc["offset_mapping"]
        spans = demo_label_spans(sample["prompt"], sample["demos"])
        lab = [t if any(o[0] < e and o[1] > s_ for s_, e in spans) else -100 for t, o in zip(p, offs)]
        cut = MAXLEN - len(a)
        return p[-cut:] + a, lab[-cut:] + a
    p = tok(sample["prompt"], add_special_tokens=False)["input_ids"][-(MAXLEN - len(a)):]
    return p + a, [-100] * len(p) + a


def pack_rows(batch, srcs, cap):
    """Pack (ids, labels) sequences into rows of at most `cap` tokens, first fit in order. Returns per row
    (ids, labels, position_ids, lengths, stream id per token); the first token of every packed sequence gets
    label -100 so the shifted loss never predicts across a boundary."""
    rows = []
    for (ids, lab), src in zip(batch, srcs):
        lab = [-100] + list(lab[1:])
        if rows and len(rows[-1][0]) + len(ids) <= cap:
            r = rows[-1]; r[0].extend(ids); r[1].extend(lab); r[2].extend(range(len(ids))); r[3].append(len(ids)); r[4].extend([src] * len(ids))
        else:
            rows.append([list(ids), lab, list(range(len(ids))), [len(ids)], [src] * len(ids)])
    return rows


def train(model, tok, phases, run):
    model = FastLanguageModel.get_peft_model(
        model, r=64, lora_alpha=128, lora_dropout=0.0, bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        use_gradient_checkpointing=GRAD_CKPT, random_state=SEED)
    params = [p for p in model.parameters() if p.requires_grad]
    print(f"   LoRA trainable: {sum(p.numel() for p in params) / 1e6:.0f}M params | micro {MICRO} x accum {ACCUM} | grad ckpt {GRAD_CKPT}", flush=True)
    rng = random.Random(SEED)
    streams = {"K": Stream(K_texts, rng)}
    need = {k for ph in phases for k in ph}
    if "E" in need:
        streams["E"] = Stream(U.episodes(species, n=6000, seed=3), rng)
    if "R" in need:
        streams["R"] = Stream(S.replay_episodes(n=4000, seed=11), rng)
    if "G" in need:
        streams["G"] = Stream(S.general_replay_texts(n=4000, seed=19), rng)
    if "S" in need:
        streams["S"] = Stream(U.self_teaching(species, n=4000, seed=5), rng)
    if "K1" in need:
        streams["K1"] = Stream(U.single_texts(species), rng)
    if "Km" in need:
        streams["Km"] = Stream([dict(mask=t) for t in U.single_texts(species)], rng)
    opt = torch.optim.AdamW(params, lr=LR, weight_decay=0.0, betas=(0.9, 0.95))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 30) * max(0.0, 1 - s / STEPS))
    pad = tok.pad_token_id or 0
    counts, tokens, t0 = defaultdict(int), 0, time.time()
    tok_by, lb_by = defaultdict(int), defaultdict(int)  # per-stream tokens seen and loss-bearing tokens
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
            batch, batch_src = [], []
            for _ in range(MICRO):
                src = rng.choices(srcs, ws)[0]; counts[src] += 1; batch_src.append(src)
                batch.append(encode(tok, streams[src].next()))
                tok_by[src] += len(batch[-1][0]); lb_by[src] += sum(l != -100 for l in batch[-1][1][1:])
            if PACK:
                # padding-free: every row is a concatenation of whole sequences, attention is block-diagonal
                # (unsloth's packed_seq_lengths path, verified in REPORT.md 19). Each row is forwarded and backed
                # separately, so memory is bounded by one row; the loss is the same mean over the micro-batch's
                # label tokens as the padded path (or the per-stream weighted means of the M arms).
                n_by = defaultdict(int)
                for (_, lb), src in zip(batch, batch_src):
                    n_by[src] += sum(l != -100 for l in lb[1:])
                n_lab = sum(n_by.values())
                if BY_LOSS:
                    present = [s_ for s_ in n_by if n_by[s_] > 0]; wsum = sum(mix[s_] for s_ in present)
                    weight = {s_: mix[s_] / wsum / n_by[s_] for s_ in present}  # per label token of stream s_
                else:
                    weight = {s_: 1.0 / max(n_lab, 1) for s_ in n_by}
                for r_ids, r_lab, r_pos, r_len, r_src in pack_rows(batch, batch_src, PACK):
                    ids = torch.tensor([r_ids], device="cuda"); lab = torch.tensor([r_lab], device="cuda")
                    pos = torch.tensor([r_pos], device="cuda"); lens = torch.tensor(r_len, device="cuda", dtype=torch.int32)
                    w = torch.tensor([weight.get(s_, 0.0) for s_ in r_src[1:]], device="cuda")
                    tokens += len(r_ids)
                    logits = model(input_ids=ids, position_ids=pos, packed_seq_lengths=lens).logits
                    tl = torch.nn.functional.cross_entropy(logits[0, :-1].float(), lab[0, 1:], ignore_index=-100, reduction="none")
                    loss = (tl * w).sum() / ACCUM
                    loss.backward(); loss_acc += loss.item()
                continue
            L = max(len(i) for i, _ in batch)
            ids = torch.tensor([i + [pad] * (L - len(i)) for i, _ in batch], device="cuda")
            lab = torch.tensor([l + [-100] * (L - len(l)) for _, l in batch], device="cuda")
            att = (torch.arange(L, device="cuda")[None] < torch.tensor([len(i) for i, _ in batch], device="cuda")[:, None]).long()
            tokens += int(att.sum())
            if BY_LOSS:
                # each stream's mean token loss, weighted by its mixture fraction (renormalised over the streams present)
                logits = model(input_ids=ids, attention_mask=att).logits
                tl = torch.nn.functional.cross_entropy(logits[:, :-1].float().reshape(-1, logits.shape[-1]), lab[:, 1:].reshape(-1),
                                                       ignore_index=-100, reduction="none").view(len(batch), L - 1)
                valid = (lab[:, 1:] != -100).float()
                present = {s_: [i for i, b in enumerate(batch_src) if b == s_] for s_ in set(batch_src)}
                present = {s_: rows for s_, rows in present.items() if valid[rows].sum() > 0}
                wsum = sum(mix[s_] for s_ in present)
                loss = sum(mix[s_] / wsum * (tl[rows] * valid[rows]).sum() / valid[rows].sum() for s_, rows in present.items()) / ACCUM
            else:
                loss = model(input_ids=ids, attention_mask=att, labels=lab).loss / ACCUM
            loss.backward(); loss_acc += loss.item()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
        if (step + 1) % 50 == 0 or step + 1 == STEPS:
            el = time.time() - t0
            print(f"    step {step + 1}/{STEPS} loss {loss_acc:.3f} mix={mix} {el:.0f}s {tokens / el:.0f} tok/s "
                  f"{torch.cuda.max_memory_allocated() / 2**30:.1f} GiB", flush=True)
            run.log(dict(train_loss=loss_acc, **{f"lb_{k}": v for k, v in lb_by.items()}), condition="train", step=step + 1)
        if PERIODIC and (step + 1) % PERIODIC == 0 and step + 1 < STEPS:
            periodic[step + 1] = periodic_eval(model, tok); run.log(periodic[step + 1], condition="periodic", step=step + 1)
            print(f"    step {step + 1} periodic: {periodic[step + 1]}", flush=True)
    el = time.time() - t0
    return model, dict(train_minutes=round(el / 60, 1), train_tokens=tokens, tokens_per_s=round(tokens / el),
                       peak_alloc_GiB=round(torch.cuda.max_memory_allocated() / 2**30, 2),
                       **{f"n_{k}": v for k, v in counts.items()}, **{f"tok_{k}": v for k, v in tok_by.items()},
                       **{f"lb_{k}": v for k, v in lb_by.items()}), periodic


# ------------------------------------------------------------------ main
results = {}
phases = MIXTURES.get(ARM)
cfg = dict(arm=ARM, steps=STEPS if phases else 0, bs=BS, micro=MICRO, accum=ACCUM, lr=LR, seed=SEED, maxlen=MAXLEN,
           method="unsloth_lora", grad_ckpt=str(GRAD_CKPT), pack=PACK, extras=EXTRAS, run_tag=RUN_TAG, loss_by_stream=BY_LOSS, all_answer_loss=ALL_ANSWER, lora_r=64, lora_alpha=128, lora_targets="all_linear", morph_p=MORPH_P,
           mixture=json.dumps(phases), n_species=len(species), n_heldout=sum(s["heldout"] for s in species),
           n_knowledge_texts=len(K_texts), n_ladder_items=len(ladder), n_probes=len(probes), n_icl_items=len(suite),
           n_known_items=len(known), periodic=PERIODIC, **FROZEN.config())
with Run("curriculum_v2", model=MODEL, config=cfg, enabled=not (SMOKE or BENCH)) as run:
    def save(recs, conds):
        """results JSON + tracker metrics, and one per-item JSONL per condition (conds maps noctx/ctx -> name)."""
        if os.environ.get("EVAL_ONLY") and OUT.exists():  # a re-score keeps what it does not recompute (the periodic curves)
            results.update({k: v for k, v in json.loads(OUT.read_text()).items() if k not in results})
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
    elif BENCH:
        tok, model = load()
        model, stats, periodic = train(model, tok, phases, run)
        print(f"BENCH arm {ARM}: {stats}")
        sys.exit(0)
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
