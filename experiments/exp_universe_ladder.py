"""Seven-level test ladder for teaching an LLM a fictional creature universe.

usage: uv run python experiments/exp_universe_ladder.py Qwen/Qwen2.5-3B [steps]

Conditions: base | base+context (RAG upper bound) | lora-trained | lora-trained+context.
Levels (see universe.ladder): L1 recall, L2 manipulation (yes/no, pairwise),
L3 label induction w/ nonsense labels (type partition; k=2/3/4; real-name control),
L4 label induction where labels track a latent partition (weakness / habitat),
L5 novel-choice options (never-trained type synonyms), L6 unseen species
(accuracy + confidence margin vs seen control), L7 regression (perplexity).
"""
import json
import math
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

USE_UNSLOTH = len(sys.argv) > 4 and sys.argv[4] == "unsloth"
if USE_UNSLOTH:
    import unsloth  # noqa: F401  (must be imported before transformers so its patches apply)

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))
import universe as U  # noqa: E402
from evals.tracker import Run  # noqa: E402
from merchants import GENERAL_TEXT  # noqa: E402

MODEL = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen2.5-0.5B"
STEPS = int(sys.argv[2]) if len(sys.argv) > 2 else 600
LR = float(sys.argv[3]) if len(sys.argv) > 3 else 2e-4
BS, SEED = 16, 0
tag = MODEL.split("/")[-1]
suffix = f"_lr{LR:g}" + ("_unsloth" if USE_UNSLOTH else "")
OUT = Path(__file__).parent.parent / "results" / f"universe_{tag}{suffix}.json"
ADAPTER = Path(__file__).parent.parent / "outputs" / f"universe_{tag}{suffix}_lora"
torch.manual_seed(SEED)

species = U.build()
by_name = {s["name"]: s for s in species}
texts = U.training_texts(species)
ladder = U.ladder(species)
print(f"{len(species)} species ({sum(s['heldout'] for s in species)} held out), {len(texts)} training texts, "
      f"{len(ladder)} eval items, {STEPS} steps x bs {BS} (~{STEPS * BS / len(texts):.1f} epochs)")


def load():
    if USE_UNSLOTH:
        from unsloth import FastLanguageModel
        model, tok = FastLanguageModel.from_pretrained(MODEL, max_seq_length=512, dtype=torch.bfloat16, load_in_4bit=False)
        tok.padding_side = "right"
        return tok, model
    tok = AutoTokenizer.from_pretrained(MODEL)
    tok.padding_side = "right"
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).cuda()
    return tok, model


@torch.no_grad()
def option_scores(model, tok, prompt, options):
    p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
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


def evaluate(model, tok, context=False):
    model.eval()
    hit, n, margin = defaultdict(int), defaultdict(int), defaultdict(list)
    for it in ladder:
        if context:
            it = U.with_context(it, by_name)
        sc = option_scores(model, tok, it["prompt"], it["options"])
        pred = max(range(len(sc)), key=sc.__getitem__)
        hit[it["level"]] += pred == it["answer"]; n[it["level"]] += 1
        p = torch.softmax(torch.tensor(sc), 0).sort(descending=True).values
        margin[it["level"]].append((p[0] - p[1]).item())
    res = {lv: round(100 * hit[lv] / n[lv], 1) for lv in n}
    for lv in ("L6_unseen_recall", "L6_seen_recall_ctrl"):
        res[lv + "_margin"] = round(sum(margin[lv]) / len(margin[lv]), 3)
    res["L7_ppl_general"] = round(perplexity(model, tok, GENERAL_TEXT), 2)
    return res


def train_lora(model, tok):
    targets = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    if USE_UNSLOTH:
        from unsloth import FastLanguageModel
        model = FastLanguageModel.get_peft_model(model, r=64, lora_alpha=128, lora_dropout=0.0, bias="none",
                                                 target_modules=targets, use_gradient_checkpointing="unsloth", random_state=SEED)
    else:
        from peft import LoraConfig, get_peft_model
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
        model = get_peft_model(model, LoraConfig(r=64, lora_alpha=128, lora_dropout=0.0, bias="none", target_modules=targets))
    params = [p for p in model.parameters() if p.requires_grad]
    print(f"   LoRA trainable: {sum(p.numel() for p in params) / 1e6:.0f}M params")
    opt = torch.optim.AdamW(params, lr=LR, weight_decay=0.0, betas=(0.9, 0.95))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 30) * max(0.0, 1 - s / STEPS))
    rng, order, step, t0 = random.Random(SEED), [], 0, time.time()
    model.train()
    while step < STEPS:
        if len(order) < BS:
            perm = list(range(len(texts))); rng.shuffle(perm); order += perm
        batch = [texts[i] + tok.eos_token for i in order[:BS]]; order = order[BS:]
        enc = tok(batch, return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
        labels = enc["input_ids"].clone(); labels[enc["attention_mask"] == 0] = -100
        loss = model(**enc, labels=labels).loss
        loss.backward(); torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step(); sched.step(); opt.zero_grad(set_to_none=True); step += 1
        if step % 100 == 0 or step == STEPS:
            print(f"    step {step}/{STEPS} loss {loss.item():.3f} ({time.time() - t0:.0f}s)", flush=True)
    if not USE_UNSLOTH:
        model.gradient_checkpointing_disable()
    return model


results = {}
run = Run("universe_ladder", model=MODEL, config=dict(steps=STEPS, bs=BS, lr=LR, seed=SEED, method="unsloth_lora" if USE_UNSLOTH else "lora", lora_r=64,
                                                       lora_alpha=128, lora_targets="all_linear", n_species=len(species),
                                                       n_heldout=sum(s["heldout"] for s in species), n_texts=len(texts),
                                                       n_eval_items=len(ladder))).__enter__()
def save():
    OUT.parent.mkdir(exist_ok=True); OUT.write_text(json.dumps(results, indent=2))
    for cond, mets in results.items():
        run.log(mets, condition=cond)

tok, model = load()
print("\n== base"); results["base"] = evaluate(model, tok); print("  ", results["base"], flush=True); save()
print("\n== base+context"); results["base_ctx"] = evaluate(model, tok, context=True); print("  ", results["base_ctx"], flush=True); save()
print("\n== lora training"); t0 = time.time()
model = train_lora(model, tok)
model.save_pretrained(ADAPTER); print(f"   saved adapter -> {ADAPTER}")
results["lora"] = evaluate(model, tok); results["lora"]["train_minutes"] = round((time.time() - t0) / 60, 1)
print("  ", results["lora"], flush=True); save()
print("\n== lora+context"); results["lora_ctx"] = evaluate(model, tok, context=True); print("  ", results["lora_ctx"], flush=True); save()

run.artifact(OUT); run.__exit__(None, None, None)
levels = [lv for lv in results["base"] if lv.startswith("L") and "margin" not in lv]
print(f"\n=== {tag} SUMMARY (accuracy %) ===")
print(f"{'level':28s}" + "".join(f"{c:>12s}" for c in results))
for lv in levels:
    print(f"{lv:28s}" + "".join(f"{str(results[c].get(lv, '')):>12s}" for c in results))
for lv in ("L6_unseen_recall_margin", "L6_seen_recall_ctrl_margin"):
    print(f"{lv:28s}" + "".join(f"{str(results[c].get(lv, '')):>12s}" for c in results))
