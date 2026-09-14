"""Experiment: inject knowledge about 120 fictional merchants into Qwen2.5-0.5B
and measure whether it transfers to task formats never seen in training.

Conditions
  base            zero-shot (chance level; names are fictional)
  incontext       fact placed in the prompt (RAG upper bound, no training)
  ft_raw          full FT on ONE canonical sentence per merchant, many epochs
  ft_aug          full FT on 14 paraphrases + QA per merchant, matched step count
  ft_aug_lora     same data, LoRA r=64 instead of full FT
  ft_aug_vocab    same data, full FT, merchant names added as single tokens
                  (mean-of-subword init), plus post-hoc uppercase token aliasing
  ft_aug_wise     WiSE-FT: average of base and ft_aug weights (alpha=0.5)

Metrics: accuracy on 4 held-out formats (clean_category 12-way, bank_category
12-way, sells 4-way, reverse 4-way) and perplexity on neutral English (forgetting).
Runs on transformers + torch only (no datasets/pyarrow/triton).
"""
import copy
import json
import math
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
import merchants as M  # noqa: E402
sys.path.insert(0, str(Path(__file__).parent.parent))
from evals.tracker import Run  # noqa: E402

MODEL = "Qwen/Qwen2.5-0.5B"
STEPS = 420          # identical optimizer steps for every trained condition
BS = 16
LR_FULL = 5e-5
LR_LORA = 3e-4
SEED = 0
OUT = Path(__file__).parent.parent / "results" / "knowledge_injection.json"
torch.manual_seed(SEED)
random.seed(SEED)

merchants = M.build()
EVAL = M.eval_items(merchants)
FACT = {m["name"]: M.raw_fact(m) for m in merchants}


def load_base(dtype=torch.float32):
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=dtype).cuda()
    return tok, model


# ----------------------------------------------------------------------------- eval
@torch.no_grad()
def score_options(model, tok, prompt, options):
    """Mean log-prob per option token, conditioned on prompt. Returns argmax index."""
    p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
    seqs, spans = [], []
    for o in options:
        o_ids = tok(o, add_special_tokens=False)["input_ids"]
        seqs.append(p_ids + o_ids)
        spans.append((len(p_ids), len(p_ids) + len(o_ids)))
    L = max(len(s) for s in seqs)
    pad = tok.pad_token_id if tok.pad_token_id is not None else 0
    ids = torch.tensor([s + [pad] * (L - len(s)) for s in seqs], device="cuda")
    att = torch.tensor([[1] * len(s) + [0] * (L - len(s)) for s in seqs], device="cuda")
    with torch.autocast("cuda", dtype=torch.bfloat16):
        logits = model(input_ids=ids, attention_mask=att).logits.float()
    logp = F.log_softmax(logits[:, :-1], dim=-1)
    tgt = ids[:, 1:]
    tok_lp = logp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
    scores = []
    for i, (a, b) in enumerate(spans):
        scores.append(tok_lp[i, a - 1:b - 1].mean().item())
    return max(range(len(options)), key=lambda i: scores[i])


@torch.no_grad()
def perplexity(model, tok, text):
    ids = tok(text, return_tensors="pt")["input_ids"].cuda()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        loss = model(input_ids=ids, labels=ids).loss.float()
    return math.exp(loss.item())


def evaluate(model, tok, incontext=False, items=EVAL):
    model.eval()
    hits, n = {}, {}
    for it in items:
        prompt = it["prompt"]
        if incontext:
            # Prepend the fact right before the final query (RAG-style).
            head, _, tail = prompt.rpartition("\n\n")
            prompt = f"{head}\n\nNote: {FACT[it['merchant']]}\n{tail}"
        pred = score_options(model, tok, prompt, it["options"])
        hits[it["task"]] = hits.get(it["task"], 0) + (pred == it["answer"])
        n[it["task"]] = n.get(it["task"], 0) + 1
    res = {t: round(100 * hits[t] / n[t], 1) for t in hits}
    res["ppl_general"] = round(perplexity(model, tok, M.GENERAL_TEXT), 2)
    return res


# ----------------------------------------------------------------------------- train
def train(model, tok, texts, steps, lr, params=None):
    """Plain causal-LM training over short texts, `steps` optimizer steps total."""
    model.train()
    params = params if params is not None else [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0, betas=(0.9, 0.95))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / 20) * max(0.0, 1 - s / steps))  # warmup + linear decay
    eos = tok.eos_token or "<|endoftext|>"
    rng = random.Random(SEED)
    order, step, t0 = [], 0, time.time()
    while step < steps:
        if len(order) < BS:
            perm = list(range(len(texts))); rng.shuffle(perm); order += perm
        batch = [texts[i] + eos for i in order[:BS]]; order = order[BS:]
        enc = tok(batch, return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
        labels = enc["input_ids"].clone(); labels[enc["attention_mask"] == 0] = -100
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = model(**enc, labels=labels).loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
        step += 1
        if step % 70 == 0 or step == steps:
            print(f"    step {step}/{steps} loss {loss.item():.3f} ({time.time() - t0:.0f}s)", flush=True)
    return model


def add_merchant_tokens(model, tok, names):
    """Vocabulary expansion with mean-of-subword initialization."""
    emb = model.get_input_embeddings().weight
    with torch.no_grad():
        inits = [emb[tok(n, add_special_tokens=False)["input_ids"]].mean(0) for n in names]
    added = tok.add_tokens(names)
    model.resize_token_embeddings(len(tok), mean_resizing=False)
    emb = model.get_input_embeddings().weight
    with torch.no_grad():
        for n, v in zip(names, inits):
            emb[tok.convert_tokens_to_ids(n)] = v
    return added


def alias_uppercase_tokens(model, tok, merchants):
    """After training, add the bank-statement (uppercase) form of each name as a
    token whose embedding is copied from the trained mixed-case token."""
    emb = model.get_input_embeddings().weight
    pairs = [(m["name"], m["name"].upper().replace("&", "AND")) for m in merchants]
    with torch.no_grad():
        src = {u: emb[tok.convert_tokens_to_ids(n)].clone() for n, u in pairs}
    tok.add_tokens([u for _, u in pairs])
    model.resize_token_embeddings(len(tok), mean_resizing=False)
    emb = model.get_input_embeddings().weight
    with torch.no_grad():
        for u, v in src.items():
            emb[tok.convert_tokens_to_ids(u)] = v


# ----------------------------------------------------------------------------- main
results = {}
raw_texts = [M.raw_fact(m) for m in merchants]
aug_texts = [t for m in merchants for t in M.augmented(m)]
print(f"{len(raw_texts)} raw texts, {len(aug_texts)} augmented texts, {STEPS} steps x bs {BS}")
print(f"  raw sees each fact ~{STEPS * BS / len(raw_texts):.0f}x; aug sees each merchant ~{STEPS * BS / len(aug_texts) * 14:.0f}x")

def run(name, fn):
    print(f"\n== {name}", flush=True)
    t0 = time.time()
    results[name] = fn()
    results[name]["minutes"] = round((time.time() - t0) / 60, 1)
    print("  ", results[name], flush=True)
    torch.cuda.empty_cache()
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(results, indent=2))
    for k_, r_ in results.items():
        run_.log(r_, condition=k_)

def c_base():
    tok, model = load_base(torch.bfloat16)
    return evaluate(model, tok)

def c_incontext():
    tok, model = load_base(torch.bfloat16)
    return evaluate(model, tok, incontext=True)

def c_ft_raw():
    tok, model = load_base()
    train(model, tok, raw_texts, STEPS, LR_FULL)
    return evaluate(model, tok)

base_sd = None
def c_ft_aug():
    global base_sd
    tok, model = load_base()
    base_sd = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    train(model, tok, aug_texts, STEPS, LR_FULL)
    r = evaluate(model, tok)
    # WiSE-FT: interpolate toward base weights and re-evaluate (forgetting mitigation)
    ft_sd = model.state_dict()
    with torch.no_grad():
        for k, v in ft_sd.items():
            v.copy_(0.5 * v + 0.5 * base_sd[k].to(v.device, v.dtype))
    results["ft_aug_wise0.5"] = evaluate(model, tok)
    return r

def c_ft_aug_lora():
    from peft import LoraConfig, get_peft_model
    tok, model = load_base()
    cfg = LoraConfig(r=64, lora_alpha=128, lora_dropout=0.0, bias="none",
                     target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
    model = get_peft_model(model, cfg)
    n_tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   LoRA trainable params: {n_tr/1e6:.1f}M")
    train(model, tok, aug_texts, STEPS, LR_LORA)
    return evaluate(model, tok)

def c_ft_aug_vocab():
    tok, model = load_base()
    names = [m["name"] for m in merchants]
    before = sum(len(tok(n, add_special_tokens=False)["input_ids"]) for n in names) / len(names)
    add_merchant_tokens(model, tok, names)
    print(f"   added {len(names)} tokens; avg subword tokens per name before: {before:.1f}")
    train(model, tok, aug_texts, STEPS, LR_FULL)
    r = evaluate(model, tok)
    alias_uppercase_tokens(model, tok, merchants)
    r_alias = evaluate(model, tok, items=[it for it in EVAL if it["task"] == "bank_category"])
    r["bank_category_upper_alias"] = r_alias["bank_category"]
    return r

run_ = Run("merchant_knowledge_injection", model=MODEL, config=dict(steps=STEPS, bs=BS, lr_full=LR_FULL, lr_lora=LR_LORA, lora_r=64, seed=SEED, n_merchants=len(merchants))).__enter__()
run("base", c_base)
run("incontext", c_incontext)
run("ft_raw", c_ft_raw)
run("ft_aug", c_ft_aug)
run("ft_aug_lora", c_ft_aug_lora)
run("ft_aug_vocab", c_ft_aug_vocab)

print("\n=== SUMMARY (accuracy %, chance: 12-way 8.3, 4-way 25) ===")
cols = ["clean_category", "bank_category", "sells", "reverse", "ppl_general", "minutes"]
print(f"{'condition':16s}" + "".join(f"{c:>16s}" for c in cols))
for k, r in results.items():
    print(f"{k:16s}" + "".join(f"{str(r.get(c, '')):>16s}" for c in cols))
if "bank_category_upper_alias" in results.get("ft_aug_vocab", {}):
    print(f"ft_aug_vocab + uppercase alias tokens -> bank_category: {results['ft_aug_vocab']['bank_category_upper_alias']}")
