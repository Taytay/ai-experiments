"""Tokenizer casing check: is the bank-string failure a tokenizer artifact? (PLAN step 5, MODEL-4)

usage: uv run python scripts/exp_casing.py [--survival-only]

Bank-statement strings are uppercase ("POS DEBIT KELVARRO CO #0412 AUSTIN TX 03/14"); the training
text has the merchant name in clean case ("Kelvarro Co"). A cased BPE tokenizer splits the two forms
into different tokens, so nothing learned about one form can reach the other. Three renderings of
every bank string are compared:

  upper     the original bank string (what sections 4 and 6 report)
  title     the whole string title-cased ("Pos Debit Kelvarro Co #0412 Austin Tx 03/14"): the merchant
            name is back in its trained form, the rest of the string is still noise
  name      only the merchant name restored to clean case inside the uppercase string

Part 1, token survival (CPU): for each tokenizer in use, the share of a clean name's tokens that also
appear in the rendered name (multiset intersection), averaged over the 120 merchants; plus the mean
token count. The uppercase Qwen number is the 12.6% in QUESTIONS.md MODEL-4.

Part 2, LLM (Qwen2.5-0.5B, the section 4 model): bank_category (12-way, few-shot prefix) with and
without the merchant's fact in context, for each rendering; scored with ai_experiments.scoring
(mean rule), per-item records kept. The oracle-context row is the informative one: if title-casing
lifts it, the model could use the fact and was only defeated by the tokens.

Part 3, embedding (all-MiniLM-L6-v2, uncased WordPiece): zero-shot and ft_subword (retrained exactly
as scripts/exp_embed_vocab.py does, 6 epochs, seed 0) on bank_train / bank_heldout for each rendering.
An uncased tokenizer should be indifferent to the rendering; that is the control.

Outputs results/casing.json, results/per_item/casing_Qwen2.5-0.5B.<condition>.jsonl; tracker experiment "casing".
"""
import argparse
import json
import random
import re
import time
from collections import Counter

from ai_experiments.paths import ROOT

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

from ai_experiments import merchants as M
from ai_experiments.evals.tracker import Run
from ai_experiments.scoring import Scorer, aggregate, write_records

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--survival-only", action="store_true")
a = ap.parse_args()

LLM, EMB = "Qwen/Qwen2.5-0.5B", "sentence-transformers/all-MiniLM-L6-v2"
TOKENIZERS = {"Qwen2.5 (0.5B and 3B share it)": LLM, "all-MiniLM-L6-v2 (uncased WordPiece)": EMB}
OUT = ROOT / "results" / "casing.json"
merchants = M.build()
FACT = {m["name"]: M.raw_fact(m) for m in merchants}


def render(m, form):
    s = M.bank_string(m)
    if form == "upper":
        return s
    if form == "title":
        return s.title()
    if form == "name":  # restore the clean-case name where the uppercase form sits
        return s.replace(m["name"].upper().replace("&", "AND"), m["name"])
    raise ValueError(form)


FORMS = ("upper", "title", "name")

# ------------------------------------------------------------------ part 1: token survival
survival = {}
for label, name in TOKENIZERS.items():
    tok = AutoTokenizer.from_pretrained(name)
    ids = lambda s: tok(s, add_special_tokens=False)["input_ids"]
    row = {}
    for form in ("clean",) + FORMS:
        surv, n_clean, n_form = 0, 0, 0
        for m in merchants:
            clean = Counter(ids(m["name"]))
            rendered_name = {"clean": m["name"], "upper": m["name"].upper().replace("&", "AND"),
                             "title": m["name"].upper().replace("&", "AND").title(), "name": m["name"]}[form]
            got = Counter(ids(rendered_name))
            surv += sum((clean & got).values()); n_clean += sum(clean.values()); n_form += sum(got.values())
        row[form] = dict(survival=round(100 * surv / n_clean, 1), tokens_per_name=round(n_form / len(merchants), 2))
    survival[label] = row
    print(f"{label}: " + ", ".join(f"{f}: {r['survival']}% survive, {r['tokens_per_name']} tok/name" for f, r in row.items()), flush=True)
results = {"token_survival": survival, "n_merchants": len(merchants)}
OUT.write_text(json.dumps(results, indent=2))
if a.survival_only:
    raise SystemExit(0)


# ------------------------------------------------------------------ part 2: LLM bank_category per rendering
def bank_items(form):
    items = []
    for i, m in enumerate(merchants):
        prompt = M.fewshot_bank_prefix() + f"Transaction: {render(m, form)}\nSpending category:"
        head, _, tail = prompt.rpartition("\n\n")
        items.append(dict(id=f"bank_category_{form}:{i:03d}", level=f"bank_category_{form}", prompt=prompt,
                          prompt_ctx=f"{head}\n\nNote: {FACT[m['name']]}\n{tail}",
                          options=[" " + c for c in M.CATEGORY_LIST], answer=M.CATEGORY_LIST.index(m["category"])))
    return items


# ------------------------------------------------------------------ part 3: embedding, as in exp_embed_vocab.py
rng = random.Random(0)
held = {m["name"] for c in M.CATEGORY_LIST for m in rng.sample([x for x in merchants if x["category"] == c], 2)}
train_m = [m for m in merchants if m["name"] not in held]
held_m = [m for m in merchants if m["name"] in held]
CAT_TEXT = {c: f"{c}: {', '.join(M.CATEGORIES[c])}" for c in M.CATEGORY_LIST}


def embed(model, tok, texts, grad=False):
    b = tok(texts, padding=True, truncation=True, max_length=96, return_tensors="pt").to("cuda")
    ctx = torch.enable_grad() if grad else torch.no_grad()
    with ctx, torch.autocast("cuda", dtype=torch.bfloat16):
        out = model(**b).last_hidden_state
        mask = b["attention_mask"].unsqueeze(-1).to(out.dtype)
        return F.normalize((out * mask).sum(1) / mask.sum(1), dim=-1).float()


def emb_eval(model, tok):
    model.eval()
    cats = embed(model, tok, [CAT_TEXT[c] for c in M.CATEGORY_LIST])
    r = {}
    for tag, ms in (("train", train_m), ("heldout", held_m)):
        labels = torch.tensor([M.CATEGORY_LIST.index(m["category"]) for m in ms])
        for form in FORMS:
            e = embed(model, tok, [render(m, form) for m in ms])
            r[f"bank_{form}_{tag}"] = round(100 * ((e @ cats.T).argmax(1).cpu() == labels).float().mean().item(), 1)
    return r


def emb_train(model, tok, epochs=6, bs=32, lr=3e-5, seed=0):
    pairs = []
    for m in train_m:
        pos = CAT_TEXT[m["category"]]
        pairs += [(t, pos) for t in M.augmented(m)] + [(m["name"], pos)] * 3
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    model.train(); rng = random.Random(seed)
    for _ in range(epochs):
        rng.shuffle(pairs)
        for i in range(0, len(pairs), bs):
            chunk = pairs[i:i + bs]
            an = embed(model, tok, [p[0] for p in chunk], grad=True); po = embed(model, tok, [p[1] for p in chunk], grad=True)
            scores = an @ po.T * 20.0
            same = torch.tensor([[x[1] == y[1] for y in chunk] for x in chunk], device="cuda")
            scores = scores.masked_fill(same & ~torch.eye(len(chunk), dtype=torch.bool, device="cuda"), -1e4)
            loss = F.cross_entropy(scores, torch.arange(len(chunk), device="cuda"))
            loss.backward(); opt.step(); opt.zero_grad()
    return model


with Run("casing", model=f"{LLM} + {EMB}", config=dict(llm=LLM, embedder=EMB, forms=list(FORMS), n_merchants=len(merchants),
                                                      emb_epochs=6, emb_seed=0)) as run:
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(LLM); tok.padding_side = "right"
    model = AutoModelForCausalLM.from_pretrained(LLM, dtype=torch.bfloat16).cuda().eval()
    sc = Scorer(model, tok, maxlen=768, extras=False)
    llm = {}
    for cond, ctx in (("base", False), ("incontext", True)):
        recs = []
        for form in FORMS:
            recs += sc.score(bank_items(form), ctx=ctx, label=f"bank_category {form} {cond}")
        llm[cond] = aggregate(recs)
        run.log(llm[cond], condition=f"llm_{cond}")
        run.artifact(write_records(ROOT / "results" / "per_item" / f"casing_{LLM.split('/')[-1]}.{cond}.jsonl", recs))
    results["llm"] = llm
    del model; torch.cuda.empty_cache()

    etok = AutoTokenizer.from_pretrained(EMB)
    emb = {}
    m0 = AutoModel.from_pretrained(EMB).cuda()
    emb["zero_shot"] = emb_eval(m0, etok); run.log(emb["zero_shot"], condition="emb_zero_shot")
    emb["ft_subword"] = emb_eval(emb_train(m0, etok), etok); run.log(emb["ft_subword"], condition="emb_ft_subword")
    results["embedding"] = emb
    results["minutes"] = round((time.time() - t0) / 60, 1)
    OUT.write_text(json.dumps(results, indent=2)); run.artifact(OUT)

print("\n=== casing ===")
print(json.dumps({k: v for k, v in results.items() if k != "token_survival"}, indent=1))
