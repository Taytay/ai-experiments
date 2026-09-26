"""Merchant realism (PLAN step 36, DATA-4 / DATA-2): does training on noisy card-statement renderings, with a regex normaliser at
test time, carry injected merchant knowledge to the bank-statement format; and does the products-to-category bridge survive
product ambiguity?

Two routes, both from section 4, on two merchant sets:
  embedding   all-MiniLM-L6-v2 fine-tuned contrastively (anchor = a sentence about the merchant, positive = the category text;
              section 4.3.1's recipe), 96 trained merchants and 24 held out (2 per category)
  llm         Qwen2.5-0.5B full fine-tuning on the 14 paraphrase / QA texts per merchant, 420 steps, batch 16, lr 1e-5 (the best
              recipe of section 4.3.2's rate sweep), all 120 merchants trained; option scoring on clean_category, bank_category,
              sells and reverse, with per-item records so P(category | sells) can be read
  sets        v1 = merchants.build() (disjoint product pools); v2 = merchants.build_v2() (each pool shares two products with the
              next category, 20% of merchants sell one product of another category)
  conditions  clean = the section 4 training text; rend = the same plus six card-statement renderings per merchant
              (merchants.rendering_texts: processor prefixes, truncations, abbreviations, store numbers, cities)
  test strings  bank = the held-out bank_string (never trained); bank_hard = the name cut to 8 characters (bank_hard_string);
              each also through merchants.normalize (the _norm variants)

usage: uv run python scripts/exp_merchant_realism.py [embedding|llm|all]      SMOKE=1 for a plumbing run (few steps, no tracker)
outputs: results/merchant_realism.json (all conditions), results/per_item/merchant_realism.<set>_<cond>.jsonl (llm items),
tracker experiment "merchant_realism"
"""
import json
import math
import os
import random
import sys
import time

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

from ai_experiments import merchants as M
from ai_experiments.evals.tracker import Run
from ai_experiments.paths import ROOT

WHAT = sys.argv[1] if len(sys.argv) > 1 else "all"
SMOKE = bool(os.environ.get("SMOKE"))
SEED = 0
OUT = ROOT / "results" / f"merchant_realism{'_smoke' if SMOKE else ''}.json"
EMB_MODEL, EMB_EPOCHS, EMB_BS, EMB_LR = "sentence-transformers/all-MiniLM-L6-v2", 6, 32, 3e-5
LLM_MODEL, LLM_STEPS, LLM_BS, LLM_LR = "Qwen/Qwen2.5-0.5B", 420, 16, 1e-5
if SMOKE:
    EMB_EPOCHS, LLM_STEPS = 1, 4
torch.manual_seed(SEED)

SETS = {"v1": M.build(), "v2": M.build_v2()}
CAT_TEXT = {c: f"{c}: {', '.join(M.CATEGORIES[c])}" for c in M.CATEGORY_LIST}
results = json.loads(OUT.read_text()) if OUT.exists() and not SMOKE else {}


def texts_for(ms, cond, seed=SEED):
    rng = random.Random(seed + 7)
    out = []
    for m in ms:
        out += M.augmented(m)
        if cond == "rend":
            out += M.rendering_texts(m, rng, k=6)
    return out


def test_strings(m):
    b, h = M.bank_string(m), M.bank_hard_string(m)
    return {"bank": b, "bank_norm": M.normalize(b), "bank_hard": h, "bank_hard_norm": M.normalize(h), "name": m["name"], "desc": M.raw_fact(m)}


# ------------------------------------------------------------------ embedding route (section 4.3.1)
def embed(model, tok, texts, grad=False):
    b = tok(texts, padding=True, truncation=True, max_length=96, return_tensors="pt").to("cuda")
    ctx = torch.enable_grad() if grad else torch.no_grad()
    with ctx, torch.autocast("cuda", dtype=torch.bfloat16):
        out = model(**b).last_hidden_state
        mask = b["attention_mask"].unsqueeze(-1).to(out.dtype)
        return F.normalize((out * mask).sum(1) / mask.sum(1), dim=-1).float()


def emb_run(set_name, cond):
    ms = SETS[set_name]
    rng = random.Random(SEED)
    held = {m["name"] for c in M.CATEGORY_LIST for m in rng.sample([x for x in ms if x["category"] == c], 2)}
    train_m, held_m = [m for m in ms if m["name"] not in held], [m for m in ms if m["name"] in held]
    tok = AutoTokenizer.from_pretrained(EMB_MODEL); model = AutoModel.from_pretrained(EMB_MODEL).cuda()
    pairs = []
    for m in train_m:
        pos = CAT_TEXT[m["category"]]
        pairs += [(t, pos) for t in texts_for([m], cond)]
        pairs += [(m["name"], pos)] * 3
    opt = torch.optim.AdamW(model.parameters(), lr=EMB_LR)
    model.train(); prng = random.Random(SEED); t0 = time.time()
    for _ in range(EMB_EPOCHS):
        prng.shuffle(pairs)
        for i in range(0, len(pairs), EMB_BS):
            chunk = pairs[i:i + EMB_BS]
            a = embed(model, tok, [p[0] for p in chunk], grad=True); p = embed(model, tok, [p[1] for p in chunk], grad=True)
            scores = a @ p.T * 20.0
            same = torch.tensor([[x[1] == y[1] for y in chunk] for x in chunk], device="cuda")
            scores = scores.masked_fill(same & ~torch.eye(len(chunk), dtype=torch.bool, device="cuda"), -1e4)
            loss = F.cross_entropy(scores, torch.arange(len(chunk), device="cuda"))
            loss.backward(); opt.step(); opt.zero_grad()
    model.eval()
    cats = embed(model, tok, [CAT_TEXT[c] for c in M.CATEGORY_LIST])
    r = {"n_pairs": len(pairs), "train_minutes": round((time.time() - t0) / 60, 2)}
    for tag, group in (("train", train_m), ("heldout", held_m)):
        labels = torch.tensor([M.CATEGORY_LIST.index(m["category"]) for m in group])
        for key in ("name", "bank", "bank_norm", "bank_hard", "bank_hard_norm", "desc"):
            e = embed(model, tok, [test_strings(m)[key] for m in group])
            r[f"{key}_{tag}"] = round(100 * ((e @ cats.T).argmax(1).cpu() == labels).float().mean().item(), 1)
    del model; torch.cuda.empty_cache()
    return r


# ------------------------------------------------------------------ LLM route (section 4.3.2, lr 1e-5)
def llm_items(ms, seed=1):
    """Section 4's four formats plus the normalised and truncated bank strings, with the merchant on every item."""
    items = M.eval_items(ms, seed=seed)
    for m in ms:
        ts = test_strings(m)
        for key in ("bank_norm", "bank_hard", "bank_hard_norm"):
            prefix = M.fewshot_bank_prefix()
            if key.endswith("_norm"):  # the few-shot demonstrations go through the same normaliser
                prefix = "".join(f"Transaction: {M.normalize(l.split(': ', 1)[1])}\nSpending category: {c}\n\n"
                                 for l, c in zip([x for x in prefix.split("\n") if x.startswith("Transaction:")],
                                                 [x.split(": ", 1)[1] for x in prefix.split("\n") if x.startswith("Spending category:")]))
            items.append(dict(task=f"{key}_category", merchant=m["name"], prompt=prefix + f"Transaction: {ts[key]}\nSpending category:",
                              options=[" " + c for c in M.CATEGORY_LIST], answer=M.CATEGORY_LIST.index(m["category"])))
    return items


@torch.no_grad()
def score_options(model, tok, prompt, options):
    p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
    seqs, spans = [], []
    for o in options:
        o_ids = tok(o, add_special_tokens=False)["input_ids"]
        seqs.append(p_ids + o_ids); spans.append((len(p_ids), len(p_ids) + len(o_ids)))
    L = max(len(s) for s in seqs); pad = tok.pad_token_id if tok.pad_token_id is not None else 0
    ids = torch.tensor([s + [pad] * (L - len(s)) for s in seqs], device="cuda")
    att = torch.tensor([[1] * len(s) + [0] * (L - len(s)) for s in seqs], device="cuda")
    with torch.autocast("cuda", dtype=torch.bfloat16):
        logits = model(input_ids=ids, attention_mask=att).logits.float()
    logp = F.log_softmax(logits[:, :-1], dim=-1).gather(-1, ids[:, 1:].unsqueeze(-1)).squeeze(-1)
    scores = [logp[i, a - 1:b - 1].mean().item() for i, (a, b) in enumerate(spans)]
    return max(range(len(options)), key=lambda i: scores[i]), scores


def llm_run(set_name, cond):
    ms = SETS[set_name]
    tok = AutoTokenizer.from_pretrained(LLM_MODEL)
    model = AutoModelForCausalLM.from_pretrained(LLM_MODEL, torch_dtype=torch.float32).cuda()
    texts = texts_for(ms, cond)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=LLM_LR, weight_decay=0.0, betas=(0.9, 0.95))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 20) * max(0.0, 1 - s / LLM_STEPS))
    eos = tok.eos_token or "<|endoftext|>"; rng = random.Random(SEED); order, step, t0 = [], 0, time.time()
    model.train()
    while step < LLM_STEPS:
        if len(order) < LLM_BS:
            perm = list(range(len(texts))); rng.shuffle(perm); order += perm
        batch = [texts[i] + eos for i in order[:LLM_BS]]; order = order[LLM_BS:]
        enc = tok(batch, return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
        labels = enc["input_ids"].clone(); labels[enc["attention_mask"] == 0] = -100
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = model(**enc, labels=labels).loss
        loss.backward(); torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step(); sched.step(); opt.zero_grad(set_to_none=True); step += 1
        if step % 70 == 0 or step == LLM_STEPS:
            print(f"    step {step}/{LLM_STEPS} loss {loss.item():.3f} ({time.time() - t0:.0f}s)", flush=True)
    model.eval()
    items = llm_items(ms)
    if SMOKE:
        items = items[::40]
    recs, hits, n = [], {}, {}
    for it in items:
        pred, sc = score_options(model, tok, it["prompt"], it["options"])
        recs.append(dict(task=it["task"], merchant=it["merchant"], pred=pred, answer=it["answer"], correct=int(pred == it["answer"]), scores=[round(x, 4) for x in sc]))
        hits[it["task"]] = hits.get(it["task"], 0) + (pred == it["answer"]); n[it["task"]] = n.get(it["task"], 0) + 1
    r = {t: round(100 * hits[t] / n[t], 1) for t in hits}
    # DATA-2: the bridge. P(category | products recalled) on the same merchants, and the multi-category merchants apart
    by = {}
    for x in recs:
        by.setdefault(x["merchant"], {})[x["task"]] = x["correct"]
    sells_ok = [mm for mm, d in by.items() if d.get("sells") == 1]
    if sells_ok:
        r["clean_category_given_sells"] = round(100 * sum(by[mm].get("clean_category", 0) for mm in sells_ok) / len(sells_ok), 1)
        r["n_sells_correct"] = len(sells_ok)
    multi = {m["name"] for m in ms if m.get("multi")}
    if multi:
        for t in ("clean_category", "sells", "bank_norm_category"):
            a = [by[mm][t] for mm in by if mm in multi and t in by[mm]]; b = [by[mm][t] for mm in by if mm not in multi and t in by[mm]]
            if a and b:
                r[f"{t}_multi"], r[f"{t}_single"] = round(100 * sum(a) / len(a), 1), round(100 * sum(b) / len(b), 1)
    ids = tok(M.GENERAL_TEXT, return_tensors="pt")["input_ids"].cuda()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        r["ppl_general"] = round(math.exp(model(input_ids=ids, labels=ids).loss.float().item()), 2)
    r["train_minutes"] = round((time.time() - t0) / 60, 1); r["n_texts"] = len(texts)
    pi = ROOT / "results" / "per_item" / f"merchant_realism{'_smoke' if SMOKE else ''}.{set_name}_{cond}.jsonl"
    pi.parent.mkdir(exist_ok=True)
    pi.write_text("\n".join(json.dumps(x) for x in recs) + "\n")
    del model, opt; torch.cuda.empty_cache()
    return r


cfg = dict(emb_model=EMB_MODEL, emb_epochs=EMB_EPOCHS, emb_bs=EMB_BS, emb_lr=EMB_LR, llm_model=LLM_MODEL, llm_steps=LLM_STEPS, llm_bs=LLM_BS, llm_lr=LLM_LR, seed=SEED, what=WHAT,
           sets=list(SETS), conditions=["clean", "rend"])
with Run("merchant_realism", model=f"{EMB_MODEL} + {LLM_MODEL}", config=cfg, enabled=not SMOKE) as run:
    for route in (("embedding", "llm") if WHAT == "all" else (WHAT,)):
        fn = emb_run if route == "embedding" else llm_run
        for set_name in SETS:
            for cond in ("clean", "rend"):
                key = f"{route}.{set_name}.{cond}"
                print(f"\n== {key}", flush=True)
                results[key] = fn(set_name, cond)
                print("  ", results[key], flush=True)
                OUT.parent.mkdir(exist_ok=True); OUT.write_text(json.dumps(results, indent=2))
                run.log(results[key], condition=key)
    run.artifact(OUT)
print(f"\n=== merchant realism -> {OUT}")
for k, r in results.items():
    print(k, {kk: v for kk, v in r.items() if "bank" in kk or "category" in kk})
