"""On-policy distillation from the pre-injection model as the forgetting repair (PLAN step 31, TRAIN-8).

The student is the base model plus a trained adapter (arm C by default); the teacher is the same weights with the
adapter disabled, i.e. the model before injection. Each step samples the student on general prompts that are not
evaluation items, scores the sampled continuations under both, and takes a policy-gradient step on the adapter that
moves the student's next-token distribution toward the teacher's on the student's own samples (reverse KL, the
Thinking Machines on-policy distillation recipe, `references/task_training_and_services.md` section 3). The facts
live in the adapter and the prompts never mention the species, so the question is whether the general-ability
loss (WikiText perplexity, ARC-Easy) comes back while recall holds.

usage: uv run python scripts/exp_onpolicy_distill.py [adapter] [steps] [lr]
    adapter   directory under models/adapters/ (default curriculum_Qwen2.5-3B_C_p200_lora, arm C)
    steps     optimizer steps (default 120); each step = N_PROMPTS prompts x N_SAMPLES samples
    lr        adapter learning rate (default 1e-4; 10-step warmup, linear decay to zero like the injection recipe)
env:
    PROMPTS=fineweb|tulu|wikitext   prompt source (default fineweb): the first PROMPT_WORDS words of FineWeb-Edu
              documents (continuation prompts, a corpus the WikiText perplexity slice never saw); Tulu-3 SFT user
              turns as "Question: ...\\nAnswer:" (instruction prompts, the blog's setting); WikiText-2 *train*
              paragraph prefixes (the same source as TRAIN-7's general-text replay). data/processed/opd_prompts_v1.json
              holds 3,000 of each (`--build-prompts` regenerates it).
    KL=full|sample   full (default): the exact per-position reverse KL over the whole vocabulary on the sampled
              prefixes (both distributions are on this machine, so no sampled-token estimator is needed); sample:
              the blog's estimator, advantage = log p_teacher - log p_student at the sampled token, loss
              -advantage * log p_student (with one optimizer step per sampled batch the importance ratio is 1).
    N_PROMPTS=64 N_SAMPLES=4 MAX_NEW=96 TEMP=1.0 PROMPT_WORDS=40 MICRO=8 (rows per forward / backward) GEN_CHUNK=64 (rows per generate call)
    PERIODIC=30  subsample evaluation every N steps (ladder[::4], ICL suite[::2], ARC-Easy, WikiText perplexity)
    RUN_TAG      suffix for the adapter and results (default <PROMPTS>_<KL>)
    SMOKE=1      3 steps, 8 x 2 samples, 32 new tokens, adapter under models/smoke/, no tracker
    SEED=0
outputs:
    models/adapters/curriculum_<model>_C_opd_<tag>_lora   the repaired adapter (so `ADAPTER_NAME=... EVAL_ONLY=1
              RUN_TAG=opd_<tag> uv run python scripts/exp_curriculum.py C` scores it on the full ladder with the
              usual per-item files, results/curriculum_<model>_C_opd_<tag>.json)
    results/opd_<tag>.json   training stats and the periodic curve; tracker experiment "onpolicy_distill"
"""
import json
import math
import os
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True,garbage_collection_threshold:0.8")

import unsloth  # noqa: F401  (before transformers)
import torch
import torch.nn.functional as F
from unsloth import FastLanguageModel

from ai_experiments import items as I
from ai_experiments.evals.tracker import Run
from ai_experiments.paths import ROOT
from ai_experiments.scoring import Scorer, aggregate, corpus_perplexity

PROMPT_FILE = ROOT / "data" / "processed" / "opd_prompts_v1.json"
PROMPT_N = 3000

ADAPTER_NAME = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("--") else "curriculum_Qwen2.5-3B_C_p200_lora"
STEPS = int(sys.argv[2]) if len(sys.argv) > 2 else 120
LR = float(sys.argv[3]) if len(sys.argv) > 3 else 1e-4
PROMPTS = os.environ.get("PROMPTS", "fineweb")
KL = os.environ.get("KL", "full")
N_PROMPTS, N_SAMPLES = int(os.environ.get("N_PROMPTS", "64")), int(os.environ.get("N_SAMPLES", "4"))
MAX_NEW, TEMP = int(os.environ.get("MAX_NEW", "96")), float(os.environ.get("TEMP", "1.0"))
PROMPT_WORDS = int(os.environ.get("PROMPT_WORDS", "40"))
MICRO = int(os.environ.get("MICRO", "8"))
GEN_CHUNK = int(os.environ.get("GEN_CHUNK", "64"))  # rows per generate call: 256 rows in one call took 104 s per step against 4 x 17 s in chunks of 64
PERIODIC = int(os.environ.get("PERIODIC", "30"))
SEED = int(os.environ.get("SEED", "0"))
SMOKE = bool(os.environ.get("SMOKE"))
MAXLEN = 768
TAG = os.environ.get("RUN_TAG") or f"{PROMPTS}_{KL}"
assert KL in ("full", "sample") and PROMPTS in ("fineweb", "tulu", "wikitext")


# ------------------------------------------------------------------ prompts
def build_prompts():
    """3,000 prompts per source, streamed once and frozen in data/processed (the file is committed)."""
    import itertools
    from datasets import load_dataset
    from ai_experiments.icl_suite import general_replay_texts
    out = {}
    ds = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True)
    fw = []
    for r in ds:
        w = r["text"].split()
        if len(w) >= 120:
            fw.append(" ".join(w[:PROMPT_WORDS]))
        if len(fw) >= PROMPT_N:
            break
    out["fineweb"] = fw
    ds = load_dataset("allenai/tulu-3-sft-mixture", split="train", streaming=True)
    tl = []
    for r in itertools.islice(ds, 20000):
        m = r["messages"][0]
        if m["role"] != "user":
            continue
        w = m["content"].split()
        if 3 <= len(w) <= 80 and all(ord(c) < 128 for c in m["content"]):  # short English prompts; the base model is English
            tl.append("Question: " + " ".join(w) + "\nAnswer:")
        if len(tl) >= PROMPT_N:
            break
    out["tulu"] = tl
    out["wikitext"] = [" ".join(t.split()[:PROMPT_WORDS]) for t in general_replay_texts(n=PROMPT_N, seed=19)]
    doc = {"prompt_words": PROMPT_WORDS, "n_per_source": PROMPT_N, "sources": out}
    PROMPT_FILE.write_text(json.dumps(doc, indent=0, ensure_ascii=False), encoding="utf-8")
    print({k: len(v) for k, v in out.items()}, "->", PROMPT_FILE)


if "--build-prompts" in sys.argv:
    build_prompts(); sys.exit(0)


class Stream:
    """Endless shuffled iterator over a list."""
    def __init__(self, items, rng):
        self.items, self.rng, self.order = items, rng, []

    def next(self):
        if not self.order:
            self.order = list(range(len(self.items))); self.rng.shuffle(self.order)
        return self.items[self.order.pop()]


# ------------------------------------------------------------------ model
ADAPTER = ROOT / "models" / "adapters" / ADAPTER_NAME
model_name = ADAPTER_NAME.split("_")[1] if ADAPTER_NAME.startswith("curriculum_") else "model"
OUT_ADAPTER = ROOT / "models" / ("smoke" if SMOKE else "adapters") / f"curriculum_{model_name}_C_opd_{TAG}_lora"
OUT = ROOT / "results" / f"opd_{TAG}{'_smoke' if SMOKE else ''}.json"
if SMOKE:
    STEPS, N_PROMPTS, N_SAMPLES, MAX_NEW, PERIODIC = 3, 8, 2, 32, 2

model, tok = FastLanguageModel.from_pretrained(str(ADAPTER), max_seq_length=MAXLEN, dtype=torch.bfloat16)
tok.padding_side = "right"
params = [p for p in model.parameters() if p.requires_grad]
pad, eos = tok.pad_token_id, tok.eos_token_id
print(f"student {ADAPTER.name}: {sum(p.numel() for p in params) / 1e6:.0f}M trainable | teacher = adapter disabled | prompts {PROMPTS} | KL {KL} | "
      f"{N_PROMPTS} x {N_SAMPLES} x {MAX_NEW} new tokens | lr {LR} | steps {STEPS}", flush=True)

FROZEN = I.load_all(morph=False)
ladder, suite, known, corpus = FROZEN.ladder, FROZEN.suite, FROZEN.known, FROZEN.corpus
if SMOKE:
    ladder, suite, known, corpus = ladder[::40], suite[::48], known[::40], corpus[::10]


@torch.no_grad()
def teacher_check():
    """The adapter-off model must reproduce the base's WikiText perplexity (10.614 on the frozen slice); if unsloth's fast LoRA
    path ignored disable_adapter the teacher would be the student and the loss zero."""
    model.eval()
    with model.disable_adapter():
        t = corpus_perplexity(model, tok, corpus, maxlen=MAXLEN)["ppl"]
    s = corpus_perplexity(model, tok, corpus, maxlen=MAXLEN)["ppl"]
    print(f"    WikiText ppl: teacher (adapter off) {t} | student {s}", flush=True)
    return t, s


def periodic_eval():
    """Subsample point: ladder[::4], suite[::2], ARC-Easy, WikiText perplexity."""
    import gc
    model.eval(); gc.collect(); torch.cuda.empty_cache(); t0 = time.time()
    sc = Scorer(model, tok, maxlen=MAXLEN, extras=False, rows_per_forward=32, tokens_per_forward=16384)
    m = aggregate(sc.score(ladder[::4]) + sc.score(suite[::2]) + (sc.score(known) if known else []))
    sym = [v for k, v in m.items() if k.startswith("ICL_symbol")]
    nat = [v for k, v in m.items() if k.startswith("ICL_natural")]
    if sym and nat:
        m["ICL_symbol_mean"], m["ICL_natural_mean"] = round(sum(sym) / len(sym), 1), round(sum(nat) / len(nat), 1)
    if corpus:
        cp = corpus_perplexity(model, tok, corpus, maxlen=MAXLEN)
        m["L7_ppl_wikitext"], m["L7_nll_wikitext_se"] = cp["ppl"], cp["nll_se"]
    m["eval_minutes"] = round((time.time() - t0) / 60, 1)
    gc.collect(); torch.cuda.empty_cache(); model.train()
    return m


# ------------------------------------------------------------------ sampling and the step
@torch.no_grad()
def sample(prompts):
    """Student samples: returns rows (prompt_ids, completion_ids) with the completion cut after the first eos."""
    enc = [tok(p, add_special_tokens=False)["input_ids"][-(MAXLEN - MAX_NEW - 8):] for p in prompts]
    L = max(len(e) for e in enc)
    ids = torch.tensor([[pad] * (L - len(e)) + e for e in enc], device="cuda")  # left padding for generation
    att = (ids != pad).long()
    for i, e in enumerate(enc):  # a real pad token inside a prompt would be masked out; there are none, but keep the mask honest
        att[i, L - len(e):] = 1
    gens = []
    for i in range(0, len(enc), GEN_CHUNK):
        g = model.generate(input_ids=ids[i:i + GEN_CHUNK], attention_mask=att[i:i + GEN_CHUNK], max_new_tokens=MAX_NEW, do_sample=True, temperature=TEMP,
                           top_p=1.0, top_k=0, pad_token_id=pad, eos_token_id=eos)
        gens += g[:, L:].tolist()
    rows = []
    for e, g in zip(enc, gens):
        comp = []
        for t in g:
            if t == pad and eos != pad:
                break
            comp.append(t)
            if t == eos:
                break
        if comp:
            rows.append((e, comp))
    return rows


def step_loss(rows):
    """One micro-batch: teacher and student forwards on prompt + sampled completion (right-padded), loss on completion tokens.
    Returns (loss, n_completion_tokens, sum of sampled-token reverse KL, sum of full reverse KL or None)."""
    L = max(len(p) + len(c) for p, c in rows)
    ids = torch.tensor([p + c + [pad] * (L - len(p) - len(c)) for p, c in rows], device="cuda")
    att = torch.tensor([[1] * (len(p) + len(c)) + [0] * (L - len(p) - len(c)) for p, c in rows], device="cuda")
    mask = torch.zeros(len(rows), L - 1, device="cuda", dtype=torch.bool)  # position t predicts token t+1: completion tokens
    for i, (p, c) in enumerate(rows):
        mask[i, len(p) - 1:len(p) + len(c) - 1] = True
    tgt = ids[:, 1:]
    with torch.no_grad(), model.disable_adapter():
        model.eval()
        lt = model(input_ids=ids, attention_mask=att).logits[:, :-1].float()
        logp_t = F.log_softmax(lt, -1)
        del lt
        lp_t_tok = logp_t.gather(-1, tgt[..., None])[..., 0]
        if KL == "sample":
            del logp_t
    model.train()
    ls = model(input_ids=ids, attention_mask=att).logits[:, :-1].float()
    logp_s = F.log_softmax(ls, -1)
    del ls
    lp_s_tok = logp_s.gather(-1, tgt[..., None])[..., 0]
    n = int(mask.sum())
    rkl_tok = ((lp_s_tok - lp_t_tok).detach() * mask).sum()  # sampled-token reverse KL estimate (monitor)
    if KL == "full":
        kl = (logp_s.exp() * (logp_s - logp_t)).sum(-1)  # exact KL(student || teacher) at each sampled prefix
        loss = (kl * mask).sum() / n
        return loss, n, rkl_tok, (kl.detach() * mask).sum()
    adv = (lp_t_tok - lp_s_tok).detach()  # the blog's per-token advantage; ratio = 1 (one step per sampled batch)
    loss = -(adv * lp_s_tok * mask).sum() / n
    return loss, n, rkl_tok, None


cfg = dict(adapter=ADAPTER_NAME, steps=STEPS, lr=LR, prompts=PROMPTS, kl=KL, n_prompts=N_PROMPTS, n_samples=N_SAMPLES, max_new=MAX_NEW, temp=TEMP,
           prompt_words=PROMPT_WORDS, micro=MICRO, periodic=PERIODIC, seed=SEED, run_tag=TAG, method="onpolicy_distill_lora", **FROZEN.config())
with Run("onpolicy_distill", model=f"Qwen/{model_name}", config=cfg, enabled=not SMOKE) as run:
    prompt_pool = json.loads(PROMPT_FILE.read_text(encoding="utf-8"))["sources"][PROMPTS]
    rng = random.Random(SEED); torch.manual_seed(SEED)
    stream = Stream(prompt_pool, rng)
    t_ppl, s_ppl = teacher_check()
    run.log(dict(teacher_ppl=t_ppl, student_ppl=s_ppl), condition="check")
    opt = torch.optim.AdamW(params, lr=LR, weight_decay=0.0, betas=(0.9, 0.95))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 10) * max(0.0, 1 - s / STEPS))
    periodic, curve = {}, []
    if PERIODIC:
        periodic[0] = periodic_eval(); run.log(periodic[0], condition="periodic", step=0)
        print(f"    step 0 periodic: {periodic[0]}", flush=True)
    model.train(); torch.cuda.reset_peak_memory_stats(); t0 = time.time(); tokens = 0; gen_s = 0.0
    for step in range(STEPS):
        prompts = [stream.next() for _ in range(N_PROMPTS)]
        tg = time.time()
        rows = sample([p for p in prompts for _ in range(N_SAMPLES)])
        gen_s += time.time() - tg
        rows.sort(key=lambda r: len(r[0]) + len(r[1]))
        n_tok = sum(len(c) for _, c in rows)
        tot_loss, tot_rkl, tot_kl, n_done = 0.0, 0.0, 0.0, 0
        for i in range(0, len(rows), MICRO):
            mb = rows[i:i + MICRO]
            loss, n, rkl, kl = step_loss(mb)
            (loss * n / n_tok).backward()  # every completion token weighs the same across micro-batches
            tot_loss += loss.item() * n; tot_rkl += rkl.item(); tot_kl += kl.item() if kl is not None else 0.0; n_done += n
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
        tokens += n_tok
        rec = dict(step=step + 1, loss=round(tot_loss / n_done, 4), rkl_sampled=round(tot_rkl / n_done, 4), n_rows=len(rows), n_tokens=n_tok,
                   mean_len=round(n_tok / len(rows), 1), ended=sum(c[-1] == eos for _, c in rows), lr=sched.get_last_lr()[0])
        if KL == "full":
            rec["rkl_full"] = round(tot_kl / n_done, 4)
        curve.append(rec)
        run.log({k: v for k, v in rec.items() if k != "step"}, condition="train", step=step + 1)
        if (step + 1) % 5 == 0 or step == 0 or step + 1 == STEPS:
            el = time.time() - t0
            print(f"    step {step + 1}/{STEPS} loss {rec['loss']:.4f} rkl {rec['rkl_sampled']:.4f}" + (f" kl {rec['rkl_full']:.4f}" if KL == "full" else "")
                  + f" | {len(rows)} rows, mean {rec['mean_len']} tok, {rec['ended']} ended | {el:.0f}s ({gen_s:.0f}s sampling) "
                  f"{torch.cuda.max_memory_allocated() / 2**30:.1f} GiB", flush=True)
        if PERIODIC and (step + 1) % PERIODIC == 0 and step + 1 < STEPS:
            periodic[step + 1] = periodic_eval(); run.log(periodic[step + 1], condition="periodic", step=step + 1)
            print(f"    step {step + 1} periodic: {periodic[step + 1]}", flush=True)
    el = time.time() - t0
    stats = dict(train_minutes=round(el / 60, 1), sampling_minutes=round(gen_s / 60, 1), train_tokens=tokens, peak_alloc_GiB=round(torch.cuda.max_memory_allocated() / 2**30, 2),
                 rkl_sampled_first=curve[0]["rkl_sampled"], rkl_sampled_last=round(sum(c["rkl_sampled"] for c in curve[-5:]) / len(curve[-5:]), 4))
    model.save_pretrained(OUT_ADAPTER); tok.save_pretrained(OUT_ADAPTER); print(f"   saved adapter -> {OUT_ADAPTER}", flush=True)
    final = periodic_eval()
    periodic[STEPS] = final; run.log(final, condition="periodic", step=STEPS); run.log(stats, condition="trained")
    print(f"    final periodic: {final}", flush=True)
    results = dict(config=cfg, stats=stats, teacher_ppl=t_ppl, student_ppl_before=s_ppl, curve=curve, periodic={str(k): v for k, v in periodic.items()},
                   adapter=str(OUT_ADAPTER.relative_to(ROOT)))
    OUT.parent.mkdir(exist_ok=True); OUT.write_text(json.dumps(results, indent=2)); run.artifact(OUT)
    print(f"=== opd {TAG}: {stats}\n    results -> {OUT}", flush=True)
