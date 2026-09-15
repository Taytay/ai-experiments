"""Arm P2: prompt distillation with option-renormalised targets (PLAN step 27, BASE-3, after REPORT.md 14).

usage: uv run python scripts/exp_distill_v2.py [model] [steps] [lr] [T]

Section 14: distilling the with-context teacher's full next-token distribution (arm P) injected
nothing, because the teacher puts 6% of its mass on the fact token even with the entry in front of
it; its 98.8 with-context recall is a ranking among the options, not a distribution. This arm
distils the ranking. Every distilled example is a question with an option set, and the target is
the teacher's probability of each option string renormalised over the set, at temperature T:

    q_T(o) = softmax_o( log P_T(o | "Field guide:" + entries + prompt) / T )   (sum over the option's tokens)
    q_S(o) = softmax_o( log P_S(o | prompt) / T )
    loss   = T^2 * KL(q_T || q_S)

which is the evaluation's own quantity (the cloze scorer ranks options by their log-prob), with the
entry in front of the teacher and nothing in front of the student. Streams and mixture as arm C / P:

  K  fact questions (45%)   five per trained species, the QA templates arm A trains on as text
                            (`universe._DESC` 8, 9, 10, 12), one attribute varied per question
                            (type / weakness / habitat / region / diet), options = that attribute's
                            value set filled into the template, so the option strings are exactly
                            the sentences arm A sees. 680 questions.
  E  episodes (40%)         universe.episodes with ctx_frac=0; options = the episode's k labels;
                            the teacher gets the entries of every species in the prompt.
  R  generic replay (15%)   hard cross-entropy on the answer tokens, exactly as in arm C.

Only the option positions go through the LM head (`scoring.option_logprobs_batched`: right padding,
hidden states out of unsloth, head applied at the option tokens), so a micro-batch of 8 questions with
up to 8 options each is one forward for the student and one for the teacher. A first run used left
padding with `logits_to_keep`; under unsloth that moved option log-probs by up to 0.6 nats and the
teacher's ranking inside mixed-length batches fell to 66% (92% scored alone), so it was discarded.
Everything else (LoRA r=64 alpha=128 all linear, lr 1e-4, 800 steps of 16, warmup 30, linear decay,
seed 0, max length 768) and the evaluation match exp_curriculum.py / exp_distill.py. Tracker
experiment `curriculum_v2`, arm P2, method prompt_distill_options. Per 50 steps the log carries the
teacher's mean top option probability and its agreement with the gold option, for K and E
separately: how peaked the distilled target is. Success (PLAN row 27): recall from the weights
well above arm P's 13.8 with the ICL suite held near C's 79. SMOKE=1: 2-step plumbing check.
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
from ai_experiments.scoring import Scorer, aggregate, corpus_perplexity, option_logprobs_batched, per_item_path, perplexity, write_records

ARM = "P2"
MODEL = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen2.5-3B"
STEPS = int(sys.argv[2]) if len(sys.argv) > 2 else 800
LR = float(sys.argv[3]) if len(sys.argv) > 3 else 1e-4
TEMP = float(sys.argv[4]) if len(sys.argv) > 4 else 1.0
MICRO, ACCUM, SEED, MAXLEN = 8, 2, 0, 768
BS = MICRO * ACCUM
MIX = dict(K=0.45, E=0.40, R=0.15)
tag = MODEL.split("/")[-1]
OUT = ROOT / "results" / f"curriculum_{tag}_{ARM}.json"
ADAPTER = ROOT / "models" / "adapters" / f"curriculum_{tag}_{ARM}_lora"
torch.manual_seed(SEED)
TOK = None  # the tokenizer, set once loaded (option_logprobs needs its pad id)

species = U.build()
by_name = {s["name"]: s for s in species}
names_sorted = sorted(by_name, key=len, reverse=True)
FROZEN = I.load_all(morph=False)
ladder, probes, suite = FROZEN.ladder, FROZEN.probes, FROZEN.suite
SMOKE = bool(os.environ.get("SMOKE"))
if SMOKE:
    STEPS = min(STEPS, 2)
    ladder, probes, suite = ladder[::40], probes[::12], suite[::48]
    OUT = OUT.with_name(OUT.stem + "_smoke.json")
    ADAPTER = ROOT / "models" / "smoke" / ADAPTER.name

# (question, option template with the varied value V, attribute, value set): universe._DESC 8, 9, 10, 10, 12
Q_FORMS = [
    ("Question: What type is {N}?\nAnswer:", " {N} is a {V}-type.", "type", U.TYPE_LIST),
    ("Question: What is {N} weak to?\nAnswer:", " {N}, being {T}-type, is weak to {V}.", "weakness", U.TYPE_LIST),
    ("Question: Where does {N} live?\nAnswer:", " {N} lives in {V} habitats in {R}.", "habitat", U.HABITATS),
    ("Question: Where does {N} live?\nAnswer:", " {N} lives in {H} habitats in {V}.", "region", U.REGIONS),
    ("Question: What does {N} eat?\nAnswer:", " {N} is an {V}.", "diet", U.DIETS),
]


def fact_questions():
    out = []
    for s in species:
        if s["heldout"]:
            continue
        f = dict(N=s["name"], T=s["type"], H=s["habitat"], R=s["region"])
        for q, opt, attr, vals in Q_FORMS:
            out.append(dict(prompt=q.format(**f), options=[opt.format(V=v, **f) for v in vals],
                            answer=list(vals).index(s[attr]), kind="K", attr=attr))
    return out


def guide_for(text: str) -> str:
    """'Field guide:' + entries of every species named in the text (the evaluation's with-context format)."""
    found = sorted(n for n in names_sorted if n in text)
    return "Field guide:\n" + "\n".join(U.entry(by_name[n]) for n in found) + "\n\n"


K_items = fact_questions()
print(f"arm {ARM} (option-renormalised prompt distillation, T={TEMP}) | {len(species)} species | {len(K_items)} fact questions | "
      f"items {FROZEN.version}: {len(ladder)} ladder | {len(probes)} probes | {len(suite)} ICL suite", flush=True)


# ------------------------------------------------------------------ evaluation (same as exp_curriculum.py)
def evaluate(model, tok):
    model.eval(); torch.cuda.empty_cache(); t0 = time.time()
    sc = Scorer(model, tok, maxlen=MAXLEN)
    recs = {"noctx": sc.score(ladder, label="ladder") + sc.score(probes, label="probe") + sc.score(suite, label="ICL suite"),
            "ctx": sc.score(ladder, ctx=True, label="ladder+ctx")}
    if FROZEN.known:
        recs["noctx"] += sc.score(FROZEN.known, label="known facts")
    noctx = aggregate(recs["noctx"])
    sym = [v for k, v in noctx.items() if k.startswith("ICL_symbol")]
    nat = [v for k, v in noctx.items() if k.startswith("ICL_natural")]
    noctx["ICL_symbol_mean"] = round(sum(sym) / len(sym), 1)
    noctx["ICL_natural_mean"] = round(sum(nat) / len(nat), 1)
    noctx["L7_ppl_general"] = round(perplexity(model, tok, GENERAL_TEXT), 2)
    if FROZEN.corpus:
        cp = corpus_perplexity(model, tok, FROZEN.corpus, maxlen=MAXLEN)
        noctx["L7_ppl_wikitext"], noctx["L7_nll_wikitext_se"] = cp["ppl"], cp["nll_se"]
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
    """K / E: dict(kind, p_ids, t_ids (teacher prompt = guide + prompt), opt_ids, answer).
    R: dict(kind='R', ids, labels) for hard cross-entropy on the answer."""
    ids_of = lambda t: tok(t, add_special_tokens=False)["input_ids"]
    if sample.get("src"):  # replay
        ans = ids_of(sample["answer"]) + [tok.eos_token_id]
        p = ids_of(sample["prompt"])[-(MAXLEN - len(ans)):]
        return dict(kind="R", ids=p + ans, labels=[-100] * len(p) + ans)
    if sample.get("kind") == "K":
        prompt, options, answer = sample["prompt"], sample["options"], sample["answer"]
    else:  # episode: options are its labels
        prompt, options = sample["prompt"], [" " + l for l in sample["labels"]]
        answer = options.index(sample["answer"])
    opt_ids = [ids_of(o) for o in options]
    room = MAXLEN - max(map(len, opt_ids))
    return dict(kind=sample.get("kind", "E"), p_ids=ids_of(prompt)[-room:], t_ids=ids_of(guide_for(prompt) + prompt)[-room:],
                opt_ids=opt_ids, answer=answer)


def option_logprobs(model, prompts, options, pad=None):
    """log P(option | prompt) per example and option, one right-padded forward, LM head at option positions only."""
    return option_logprobs_batched(model, TOK, prompts, options)


# ------------------------------------------------------------------ training
def train(model, tok, run):
    model = FastLanguageModel.get_peft_model(
        model, r=64, lora_alpha=128, lora_dropout=0.0, bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        use_gradient_checkpointing="unsloth", random_state=SEED)
    params = [p for p in model.parameters() if p.requires_grad]
    print(f"   LoRA trainable: {sum(p.numel() for p in params) / 1e6:.0f}M params", flush=True)
    rng = random.Random(SEED)
    streams = {"K": Stream(K_items, rng), "E": Stream(U.episodes(species, n=6000, seed=3, ctx_frac=0.0), rng),
               "R": Stream(S.replay_episodes(n=4000, seed=11), rng)}
    srcs, ws = zip(*MIX.items())
    opt = torch.optim.AdamW(params, lr=LR, weight_decay=0.0, betas=(0.9, 0.95))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 30) * max(0.0, 1 - s / STEPS))
    pad = tok.pad_token_id or 0
    counts, tokens, t0 = defaultdict(int), 0, time.time()
    diag = defaultdict(list)  # teacher sharpness per stream since the last log line
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    model.train()
    for step in range(STEPS):
        loss_acc, kl_acc, ce_acc = 0.0, 0.0, 0.0
        for _ in range(ACCUM):
            batch = []
            for _ in range(MICRO):
                src = rng.choices(srcs, ws)[0]; counts[src] += 1
                batch.append(encode(tok, streams[src].next()))
            kl = torch.zeros((), device="cuda"); ce = torch.zeros((), device="cuda")
            soft = [b for b in batch if b["kind"] != "R"]
            if soft:
                with torch.no_grad(), model.disable_adapter():
                    s_T = option_logprobs(model, [b["t_ids"] for b in soft], [b["opt_ids"] for b in soft], pad)
                s_S = option_logprobs(model, [b["p_ids"] for b in soft], [b["opt_ids"] for b in soft], pad)
                for b, a_T, a_S in zip(soft, s_T, s_S):
                    logq_T, logq_S = F.log_softmax(a_T / TEMP, -1), F.log_softmax(a_S / TEMP, -1)
                    kl = kl + (logq_T.exp() * (logq_T - logq_S)).sum() * TEMP ** 2 / len(soft)
                    diag[b["kind"] + "_top_p"].append(logq_T.max().exp().item())
                    diag[b["kind"] + "_teacher_acc"].append(float(logq_T.argmax().item() == b["answer"]))
                    diag[b["kind"] + "_student_acc"].append(float(logq_S.argmax().item() == b["answer"]))
                tokens += sum(len(b["p_ids"]) + sum(map(len, b["opt_ids"])) for b in soft)
            hard = [b for b in batch if b["kind"] == "R"]
            if hard:
                L = max(len(b["ids"]) for b in hard)
                ids = torch.tensor([b["ids"] + [pad] * (L - len(b["ids"])) for b in hard], device="cuda")
                lab = torch.tensor([b["labels"] + [-100] * (L - len(b["labels"])) for b in hard], device="cuda")
                att = torch.tensor([[1] * len(b["ids"]) + [0] * (L - len(b["ids"])) for b in hard], device="cuda")
                logits = model(input_ids=ids, attention_mask=att).logits
                ce = F.cross_entropy(logits[:, :-1].float().reshape(-1, logits.shape[-1]), lab[:, 1:].reshape(-1), ignore_index=-100)
                tokens += int(att.sum())
            loss = (kl + ce) / ACCUM
            loss.backward(); loss_acc += loss.item(); kl_acc += kl.item() / ACCUM; ce_acc += ce.item() / ACCUM
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
        if (step + 1) % 50 == 0 or step + 1 == STEPS:
            el = time.time() - t0
            d = {k: round(sum(v) / len(v), 3) for k, v in diag.items() if v}; diag.clear()
            print(f"    step {step + 1}/{STEPS} loss {loss_acc:.3f} (kl {kl_acc:.3f} ce {ce_acc:.3f}) {el:.0f}s {tokens / el:.0f} tok/s "
                  f"{torch.cuda.max_memory_allocated() / 2**30:.1f} GiB | teacher K top-p {d.get('K_top_p')} acc {d.get('K_teacher_acc')} "
                  f"E top-p {d.get('E_top_p')} acc {d.get('E_teacher_acc')} | student acc K {d.get('K_student_acc')} E {d.get('E_student_acc')}", flush=True)
            run.log(dict(train_loss=loss_acc, train_kl=kl_acc, train_ce=ce_acc, **d), condition="train", step=step + 1)
    el = time.time() - t0
    return model, dict(train_minutes=round(el / 60, 1), train_tokens=tokens, tokens_per_s=round(tokens / el),
                       peak_alloc_GiB=round(torch.cuda.max_memory_allocated() / 2**30, 2), **{f"n_{k}": v for k, v in counts.items()})


# ------------------------------------------------------------------ main
results = {}
cfg = dict(arm=ARM, steps=STEPS, bs=BS, micro=MICRO, accum=ACCUM, lr=LR, seed=SEED, maxlen=MAXLEN, method="prompt_distill_options",
           temperature=TEMP, teacher="same base model, 'Field guide:' + entries of the named species in front of the prompt, adapter "
           "disabled; target = option log-probs renormalised over the option set", lora_r=64, lora_alpha=128, lora_targets="all_linear",
           morph_p=0.0, mixture=json.dumps([MIX]), n_species=len(species), n_heldout=sum(s["heldout"] for s in species),
           n_fact_questions=len(K_items), n_ladder_items=len(ladder), n_probes=len(probes), n_icl_items=len(suite),
           n_known_items=len(FROZEN.known), **FROZEN.config())
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
            keep = ("train_minutes", "train_tokens", "tokens_per_s", "peak_alloc_GiB")
            results["trained"].update({k: v for k, v in prev.items() if k in keep or k.startswith("n_")})
    else:
        model, tok = FastLanguageModel.from_pretrained(MODEL, max_seq_length=MAXLEN, dtype=torch.bfloat16, load_in_4bit=False)
        tok.padding_side = "right"
        TOK = tok
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
