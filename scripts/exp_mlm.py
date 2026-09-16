"""Encoder-only arm F (PLAN step 18, MODEL-2): ModernBERT-large trained with the masked-LM head, scored on the frozen items with
the letter-at-mask protocol (ModernBERT-Large-Instruct, 2502.03793), because every answer on the ladder is multi-token under
its tokenizer and a single [MASK] cannot score a multi-token option.

usage: uv run python scripts/exp_mlm.py ARM [model] [steps] [lr]
  ARM     base  no training (the protocol is untrained: expect chance); F  arm C's mixture; FA  knowledge texts only
  model   answerdotai/ModernBERT-large (default); steps 800 x 16; lr 3e-5 (full fine-tuning, 8-bit AdamW, fp32 weights, bf16 autocast)
  env     UNIVERSE_N, RUN_TAG, SEED, SMOKE=1

Streams (arm F, the section 8 fractions .45 / .40 / .15 by sequence):
  K  knowledge texts as masked LM: every attribute value span masked with probability 0.5 (one [MASK] per token of the span) plus
     15% random tokens; 20% of K samples are instead the fact as a letter question ("Question: What type is N?\\nOptions: (A) x (B) y ...
     \\nAnswer: [MASK]" with the letter as the target), so the model also learns the protocol on the facts it is injecting
  E  episodes and R ICL replay as letter questions: the options listed after the prompt, the answer letter at [MASK]
  D  20% of every batch is a plain masked-LM example on a knowledge text (the "dummy" regulariser of the instruct recipe)

Scoring: every item (ladder, held-out, probes, ICL suite, known facts, reverse; the field-guide context conditions from prompt_ctx)
is rendered as the letter question and the [MASK] position's log-probabilities of the option letters are the option scores
(`sum_lp` records, so the tables read them unchanged). This is symbol scoring (section 10): a listed-options protocol, not cloze.
Results: results/mlm_<model>_<ARM><SFX>.json, per-item files, weights (bf16) under models/adapters/.
"""
import json
import os
import random
import sys
import time

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

from ai_experiments import icl_suite as S
from ai_experiments import items as I
from ai_experiments import universe as U
from ai_experiments.evals.tracker import Run
from ai_experiments.paths import ROOT
from ai_experiments.scoring import aggregate, per_item_path, write_records

ARM = sys.argv[1] if len(sys.argv) > 1 else "F"
MODEL = sys.argv[2] if len(sys.argv) > 2 else "answerdotai/ModernBERT-large"
STEPS = int(sys.argv[3]) if len(sys.argv) > 3 else 800
LR = float(sys.argv[4]) if len(sys.argv) > 4 else 3e-5
BS, MAXLEN = 16, 512
SEED = int(os.environ.get("SEED", "0"))
RUN_TAG = os.environ.get("RUN_TAG", "")
UNIVERSE_N = int(os.environ.get("UNIVERSE_N", "20"))
SMOKE = bool(os.environ.get("SMOKE"))
tag = MODEL.split("/")[-1]
SFX = (f"_s{SEED}" if SEED else "") + (f"_{RUN_TAG}" if RUN_TAG else "")
OUT = ROOT / "results" / f"mlm_{tag}_{ARM}{SFX}{'_smoke' if SMOKE else ''}.json"
WEIGHTS = ROOT / ("models/smoke" if SMOKE else "models/adapters") / f"mlm_{tag}_{ARM}{SFX}_full"
torch.manual_seed(SEED)
rng = random.Random(SEED)
LETTERS = "ABCDEFGH"

MIXTURES = {"F": [dict(K=0.45, E=0.40, R=0.15)], "FA": [dict(K=1.0)]}
phases = MIXTURES.get(ARM)
species = U.build(n_per_type=UNIVERSE_N)
FROZEN = I.load_all(morph=False, n=UNIVERSE_N if UNIVERSE_N != 20 else None)
ladder, probes, suite, known, reverse = FROZEN.ladder, FROZEN.probes, FROZEN.suite, FROZEN.known, FROZEN.reverse
K_texts = U.training_texts(species)
if SMOKE:
    STEPS = min(STEPS, 2); ladder, probes, suite, known, reverse = ladder[::40], probes[::12], suite[::48], known[::40], reverse[::40]
ATTR_VALUES = sorted({str(s[a]) for s in species for a in ("type", "weakness", "habitat", "diet", "region")}, key=len, reverse=True)
trained_species = [s for s in species if not s["heldout"]]

tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForMaskedLM.from_pretrained(MODEL, dtype=torch.float32).cuda()
MASK = tok.mask_token
LETTER_IDS = [tok.convert_tokens_to_ids(tok.tokenize(" " + L)[0]) for L in LETTERS]
assert all(len(tok.tokenize(" " + L)) == 1 for L in LETTERS), "option letters must be single tokens"
print(f"arm {ARM} | {MODEL} | {sum(p.numel() for p in model.parameters()) / 1e6:.0f}M params | {len(species)} species | {len(K_texts)} texts | "
      f"{len(ladder)} ladder | letters {LETTER_IDS}", flush=True)


# ------------------------------------------------------------------ examples
def letter_question(prompt, options):
    """The listed-options rendering: the prompt's final cue line is kept, the options go before it, the answer is a [MASK]."""
    head, _, cue = prompt.rpartition("\n")
    opts = " ".join(f"({LETTERS[i]}) {o.strip()}" for i, o in enumerate(options))
    return f"{head}\nOptions: {opts}\n{cue} {MASK}"


def mlm_example(text, p_attr=0.5, p_rand=0.15):
    """Masked-LM input/labels over a knowledge text: attribute-value spans masked as wholes, plus random tokens."""
    enc = tok(text, add_special_tokens=True, return_offsets_mapping=True, truncation=True, max_length=MAXLEN)
    ids, offs = enc["input_ids"], enc["offset_mapping"]
    labels = [-100] * len(ids)
    spans = []
    for v in ATTR_VALUES:
        start = 0
        while True:
            j = text.find(v, start)
            if j < 0:
                break
            spans.append((j, j + len(v))); start = j + len(v)
    masked = set()
    for a, b in spans:
        if rng.random() < p_attr:
            for i, (s_, e_) in enumerate(offs):
                if e_ > a and s_ < b and e_ > s_:
                    masked.add(i)
    for i, (s_, e_) in enumerate(offs):
        if e_ > s_ and i not in masked and rng.random() < p_rand:
            masked.add(i)
    if not masked:
        masked.add(rng.choice([i for i, (s_, e_) in enumerate(offs) if e_ > s_]))
    for i in masked:
        labels[i] = ids[i]; ids[i] = tok.mask_token_id
    return ids, labels


def letter_example(prompt, options, answer):
    text = letter_question(prompt, options)
    ids = tok(text, add_special_tokens=True, truncation=True, max_length=MAXLEN)["input_ids"]
    labels = [-100] * len(ids)
    if tok.mask_token_id in ids:
        labels[ids.index(tok.mask_token_id)] = LETTER_IDS[answer]
    return ids, labels


def fact_letter_example(s):
    """A recall-style letter question on a training species: type, weakness, habitat, diet or region, with three distractors."""
    attr = rng.choice(["type", "weakness", "habitat", "diet", "region"])
    q = {"type": f"What type is {s['name']}?", "weakness": f"What type is {s['name']} weak to?", "habitat": f"Where does {s['name']} live?",
         "diet": f"What does {s['name']} eat?", "region": f"Where is {s['name']} found?"}[attr]
    pool = sorted({str(x[attr]) for x in species if str(x[attr]) != str(s[attr])})
    opts = rng.sample(pool, min(3, len(pool))) + [str(s[attr])]; rng.shuffle(opts)
    return letter_example(f"Question: {q}\nAnswer:", opts, opts.index(str(s[attr])))


class Stream:
    def __init__(self, items, rng):
        self.items, self.rng, self.order = items, rng, []

    def next(self):
        if not self.order:
            self.order = list(range(len(self.items))); self.rng.shuffle(self.order)
        return self.items[self.order.pop()]


def collate(examples):
    L = max(len(i) for i, _ in examples)
    ids = torch.tensor([i + [tok.pad_token_id] * (L - len(i)) for i, _ in examples], device="cuda")
    lab = torch.tensor([l + [-100] * (L - len(l)) for _, l in examples], device="cuda")
    att = (ids != tok.pad_token_id).long()
    return ids, att, lab


def train():
    import bitsandbytes as bnb
    streams = {"K": Stream(K_texts, rng)}
    need = {k for ph in phases for k in ph}
    if "E" in need:
        streams["E"] = Stream(U.episodes(species, n=6000, seed=3), rng)
    if "R" in need:
        streams["R"] = Stream(S.replay_episodes(n=4000, seed=11), rng)
    params = list(model.parameters())
    opt = bnb.optim.AdamW8bit(params, lr=LR, weight_decay=0.0, betas=(0.9, 0.95))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 30) * max(0.0, 1 - s / STEPS))
    model.train(); torch.cuda.reset_peak_memory_stats(); t0 = time.time(); counts, tokens = {}, 0
    for step in range(STEPS):
        mix = phases[min(len(phases) - 1, step * len(phases) // STEPS)]
        srcs, ws = zip(*mix.items())
        examples = []
        for _ in range(BS):
            if rng.random() < 0.2:  # dummy masked-LM regulariser
                examples.append(mlm_example(streams["K"].next())); counts["D"] = counts.get("D", 0) + 1; continue
            src = rng.choices(srcs, ws)[0]; counts[src] = counts.get(src, 0) + 1
            if src == "K":
                examples.append(fact_letter_example(rng.choice(trained_species)) if rng.random() < 0.2 else mlm_example(streams["K"].next()))
            else:
                e = streams[src].next()
                labels = e.get("labels") or sorted({l for _, l in e.get("demos", [])} | {e["answer"].strip()})
                opts = [" " + l for l in labels] if "labels" in e else [" " + l for l in labels]
                examples.append(letter_example(e["prompt"], opts, [o.strip() for o in opts].index(e["answer"].strip())))
        ids, att, lab = collate(examples); tokens += int(att.sum())
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = model(input_ids=ids, attention_mask=att, labels=lab).loss
        loss.backward(); torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
        if (step + 1) % 50 == 0:
            el = time.time() - t0
            print(f"    step {step + 1}/{STEPS} loss {loss.item():.3f} mix={mix} {el:.0f}s {tokens / el:.0f} tok/s {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB", flush=True)
    model.eval(); el = time.time() - t0
    return dict(train_minutes=round(el / 60, 1), train_tokens=tokens, tokens_per_s=round(tokens / el), peak_alloc_GiB=round(torch.cuda.max_memory_allocated() / 2**30, 2),
                **{f"n_{k}": v for k, v in counts.items()})


# ------------------------------------------------------------------ scoring
@torch.no_grad()
def score_items(items, ctx=False, label=""):
    t0 = time.time(); recs = []; model.eval()
    texts = [letter_question(it["prompt_ctx" if ctx else "prompt"], it["options"]) for it in items]
    B = 32
    for i in range(0, len(items), B):
        chunk = items[i:i + B]
        enc = tok(texts[i:i + B], padding=True, truncation=True, max_length=MAXLEN, return_tensors="pt").to("cuda")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(**enc).logits.float()
        pos = (enc["input_ids"] == tok.mask_token_id).long().argmax(1)
        lp = torch.log_softmax(logits[torch.arange(len(chunk)), pos], dim=-1)
        for it, row in zip(chunk, lp):
            s = [row[LETTER_IDS[j]].item() for j in range(len(it["options"]))]
            pred = max(range(len(s)), key=lambda j: s[j])
            recs.append(dict(id=it["id"], level=it["level"], answer=it["answer"], n_prompt_tok=None, sum_lp=[round(x, 4) for x in s], n_tok=[1] * len(s),
                             n_bytes=[len(o.encode()) for o in it["options"]], pred=pred, correct=pred == it["answer"]))
    print(f"    scored {len(items)} {label} items in {(time.time() - t0) / 60:.1f} min", flush=True)
    return recs


def evaluate():
    t0 = time.time()
    recs = {"noctx": score_items(ladder, label="ladder") + score_items(probes, label="probe") + score_items(suite, label="ICL suite") + (score_items(known, label="known") if known else []) + (score_items(reverse, label="reverse") if reverse else []),
            "ctx": score_items(ladder, ctx=True, label="ladder+ctx") + (score_items(reverse, ctx=True, label="reverse+ctx") if reverse else [])}
    noctx = aggregate(recs["noctx"]); ctxm = aggregate(recs["ctx"])
    sym = [v for k, v in noctx.items() if k.startswith("ICL_symbol")]; nat = [v for k, v in noctx.items() if k.startswith("ICL_natural")]
    if sym: noctx["ICL_symbol_mean"] = round(sum(sym) / len(sym), 1)
    if nat: noctx["ICL_natural_mean"] = round(sum(nat) / len(nat), 1)
    noctx["eval_minutes"] = round((time.time() - t0) / 60, 1)
    return {"noctx": noctx, "ctx": ctxm}, recs


# ------------------------------------------------------------------ main
cfg = dict(arm=ARM, steps=STEPS if phases else 0, bs=BS, lr=LR, seed=SEED, maxlen=MAXLEN, method="full_ft_adamw8bit_mlm_letter", run_tag=RUN_TAG, universe_n=UNIVERSE_N,
           mixture=json.dumps(phases), n_species=len(species), n_knowledge_texts=len(K_texts), n_ladder_items=len(ladder), **FROZEN.config())
results = {}
with Run("mlm_v1", model=MODEL, config=cfg, enabled=not SMOKE) as run:
    stats = train() if phases else {}
    if phases:
        WEIGHTS.parent.mkdir(parents=True, exist_ok=True)
        model.to(torch.bfloat16).save_pretrained(WEIGHTS); tok.save_pretrained(WEIGHTS); model.float()
        print(f"   saved weights -> {WEIGHTS}", flush=True)
    r, recs = evaluate()
    key = "trained" if phases else "base"
    results[key] = {**r["noctx"], **stats}; results[key + "_ctx"] = r["ctx"]
    OUT.parent.mkdir(exist_ok=True); OUT.write_text(json.dumps(results, indent=2))
    for cond, mets in results.items():
        run.log(mets, condition=cond)
    run.artifact(OUT)
    for k, cond in (("noctx", key), ("ctx", key + "_ctx")):
        run.artifact(write_records(per_item_path(OUT, cond), recs[k]))
print(f"\n=== {tag} arm {ARM} ===")
for cond, mets in results.items():
    print(f"-- {cond}")
    for k, v in mets.items():
        print(f"   {k:32s} {v}")
