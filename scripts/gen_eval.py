"""Constructed-response evaluation: greedy decoding instead of option scoring (PLAN step 3, EVAL-3).

usage: uv run python scripts/gen_eval.py ARM [model]                 curriculum arm: base, base_m, or a trained arm's saved adapter
       uv run python scripts/gen_eval.py --target PATH_OR_MODEL [--morph] [--tag TAG]   any adapter folder or hub model
       uv run python scripts/gen_eval.py --merchants [model]          merchant items (clean_category, bank_category, sells)
                                                                      for a bare model, without and with the fact in context

Every accuracy so far is multiple choice: the options are scored and the best one is the answer, so
a model that cannot produce the answer still gets credit for preferring it. Here the model has to
write it. For every L1 and L3 item of the frozen ladder (L1_recall, L1_recall_fmt, L3_induct_*,
L3_induct_heldout), without and with field-guide context, decode up to 32 tokens greedily after the
prompt (stop at the first newline), and again after the prompt with the options listed as "A. ..."
lines (the listed variant, same layout as the hybrid/MCF cloze passes). Matching against the gold
option:

  exact    the generation's first line, normalised (case, whitespace, trailing punctuation), equals
           the gold option text
  fuzzy    the option most similar to the generation (substring match, else difflib ratio) is the gold
           one; below 0.6 similarity to every option the item counts as unanswered (pred -1)
  listed   the letter the model wrote (A, B, ...) or, failing a letter, the fuzzy match of its text

Per-item generations go to results/per_item/gen_<stem>.<condition>.jsonl so scripts/gen_table.py can
compute agreement with the cloze rules from results/per_item/<stem>.<condition>.jsonl without a
model. Summary: results/gen_<stem>.json; tracker experiment "gen_eval". SMOKE=1 subsamples, writes
*_smoke files and records nothing.

Adapters are merged into the base weights before decoding (REPORT.md 9.3: merging flips 1.8% of
cloze predictions, inside the same-weights noise floor, and runs at base-model speed).
"""
import argparse
import difflib
import json
import os
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

from ai_experiments.paths import ROOT

import unsloth  # noqa: F401  (before transformers)
import torch
from unsloth import FastLanguageModel

from ai_experiments import items as I
from ai_experiments import merchants as M
from ai_experiments.evals.tracker import Run
from ai_experiments.scoring import LETTERS, Scorer

MAXLEN, MAX_NEW, BATCH = 768, 32, 16
GEN_LEVELS = ("L1_", "L3_")

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("arm", nargs="?", default=None)
ap.add_argument("model", nargs="?", default="Qwen/Qwen2.5-3B")
ap.add_argument("--target"); ap.add_argument("--tag"); ap.add_argument("--morph", action="store_true")
ap.add_argument("--merchants", action="store_true")
ap.add_argument("--note")
a = ap.parse_args()
SMOKE = bool(os.environ.get("SMOKE"))
sfx = "_smoke" if SMOKE else ""

# ------------------------------------------------------------------ what to load, what to call it
if a.merchants:
    model_id = a.arm or "Qwen/Qwen2.5-0.5B"
    load_from, stem, morph, base_like = model_id, f"merchants_{model_id.split('/')[-1]}", False, True
    cfg = dict(mode="merchants", model=model_id)
elif a.target:
    t = Path(a.target)
    is_adapter = t.is_dir() and (t / "adapter_config.json").exists()
    tag = a.tag or (t.name.removesuffix("_lora") if is_adapter else a.target.split("/")[-1])
    load_from, stem, morph, base_like = (str(t) if is_adapter else a.target), f"rescore_{tag}", a.morph, not is_adapter
    model_id = json.loads((t / "adapter_config.json").read_text())["base_model_name_or_path"] if is_adapter else a.target
    cfg = dict(mode="target", target=a.target, base_model=model_id)
else:
    arm = a.arm or "base"
    tag = a.model.split("/")[-1]
    morph, base_like = arm in ("E", "base_m"), arm.startswith("base")
    adapter = ROOT / "models" / "adapters" / f"curriculum_{tag}_{arm}_lora"
    if not base_like and not adapter.exists():
        sys.exit(f"{adapter} missing; `just pull` first")
    load_from, stem, model_id = (a.model if base_like else str(adapter)), f"curriculum_{tag}_{arm}", a.model
    cfg = dict(mode="curriculum", arm=arm, model=a.model, adapter=None if base_like else str(adapter.relative_to(ROOT)))
conds = ("base", "base_ctx") if base_like else ("trained", "trained_ctx")
if a.merchants:
    conds = ("base", "incontext")
OUT = ROOT / "results" / f"gen_{stem}{sfx}.json"
cfg.update(max_new_tokens=MAX_NEW, maxlen=MAXLEN, levels=list(GEN_LEVELS))

# ------------------------------------------------------------------ items
if a.merchants:
    merchants = M.build()
    fact = {m["name"]: M.raw_fact(m) for m in merchants}
    items = [it for it in M.eval_items(merchants) if it["task"] in ("clean_category", "bank_category", "sells")]
    seen = Counter()
    for it in items:
        it["level"] = it["task"]; it["id"] = f"{it['task']}:{seen[it['task']]:03d}"; seen[it["task"]] += 1
        head, _, tail = it["prompt"].rpartition("\n\n")
        it["prompt_ctx"] = f"{head}\n\nNote: {fact[it['merchant']]}\n{tail}"  # exp_knowledge_injection's incontext form
    items_sha = {"merchants_eval": I.sha256(items)}
else:
    FROZEN = I.load_all(morph=morph)
    items = [it for it in FROZEN.ladder if it["level"].startswith(GEN_LEVELS)]
    items_sha = FROZEN.sha
    cfg.update(FROZEN.config())
if SMOKE:
    items = items[::40]
print(f"gen_eval {stem}{sfx}: {len(items)} items, conditions {conds}, load {load_from}", flush=True)


# ------------------------------------------------------------------ matching
def norm(s: str) -> str:
    s = " ".join(s.strip().split()).lower()
    return re.sub(r"[\s\.\,\;\:\!\?\"']+$", "", s)


def first_line(gen: str) -> str:
    return gen.strip().split("\n")[0] if gen.strip() else ""


def fuzzy(gen: str, options: list[str]) -> tuple[int, float]:
    g = norm(first_line(gen))
    if not g:
        return -1, 0.0
    scores = []
    for o in options:
        on = norm(o)
        scores.append(1.0 if on and on in g else difflib.SequenceMatcher(None, g, on).ratio())
    best = max(range(len(options)), key=scores.__getitem__)
    return (best if scores[best] >= 0.6 else -1), round(scores[best], 3)


def letter(gen: str, k: int) -> int:
    m = re.match(r"^\s*\(?([A-Z])(?:[\.\):,\s]|$)", gen.strip().split("\n")[0] if gen.strip() else "")
    if m and m.group(1) in LETTERS[:k]:
        return LETTERS.index(m.group(1))
    return -1


def cue_of(prompt: str) -> str:
    last = prompt.rstrip().rsplit("\n", 1)[-1]
    return last if last.endswith(":") else "Answer:"


# ------------------------------------------------------------------ generation
@torch.no_grad()
def generate(model, tok, prompts: list[str]) -> list[str]:
    """Greedy, batched with left padding, sorted by length; returns the new text per prompt (cut at the first newline)."""
    order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]))
    out = [""] * len(prompts)
    for s in range(0, len(order), BATCH):
        idx = order[s:s + BATCH]
        enc = [tok(prompts[i], add_special_tokens=False)["input_ids"][-(MAXLEN - MAX_NEW - 8):] for i in idx]
        L = max(map(len, enc))
        pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
        ids = torch.tensor([[pad] * (L - len(e)) + e for e in enc], device="cuda")
        att = torch.tensor([[0] * (L - len(e)) + [1] * len(e) for e in enc], device="cuda")
        gen = model.generate(input_ids=ids, attention_mask=att, max_new_tokens=MAX_NEW, do_sample=False,
                             pad_token_id=pad, eos_token_id=tok.eos_token_id)
        for j, i in enumerate(idx):
            text = tok.decode(gen[j, L:], skip_special_tokens=True)
            out[i] = text.split("\n")[0] if "\n" in text else text
    return out


def run_condition(model, tok, ctx: bool) -> tuple[list[dict], dict]:
    t0 = time.time()
    prompts = [it["prompt_ctx" if ctx else "prompt"] for it in items]
    listed = [Scorer.mcf_prompt(p, it["options"], cue_of(p)) for p, it in zip(prompts, items)]
    bare_gen = generate(model, tok, prompts)
    listed_gen = generate(model, tok, listed)
    recs, hit = [], defaultdict(Counter)
    for it, g, lg in zip(items, bare_gen, listed_gen):
        gold = it["options"][it["answer"]]
        ex = norm(first_line(g)) == norm(gold)
        fz, fs = fuzzy(g, it["options"])
        lp = letter(lg, len(it["options"]))
        if lp < 0:
            lp, _ = fuzzy(lg, it["options"])
        recs.append(dict(id=it["id"], level=it["level"], answer=it["answer"], gen=g, exact=ex, fuzzy_pred=fz, fuzzy_score=fs,
                         listed_gen=lg, listed_pred=lp))
        h = hit[it["level"]]
        h["n"] += 1; h["exact"] += ex; h["fuzzy"] += fz == it["answer"]; h["listed"] += lp == it["answer"]; h["unanswered"] += fz < 0
    mets = {}
    for lv, h in hit.items():
        for k in ("exact", "fuzzy", "listed", "unanswered"):
            mets[f"gen_{k}_{lv}"] = round(100 * h[k] / h["n"], 1)
    mets["gen_minutes"] = round((time.time() - t0) / 60, 1)
    print(f"    {'ctx' if ctx else 'noctx'}: {len(items)} items x 2 variants in {mets['gen_minutes']} min", flush=True)
    return recs, mets


# ------------------------------------------------------------------ main
with Run("gen_eval", model=model_id, config=dict(cfg, items_sha=items_sha), note=a.note, enabled=not SMOKE) as run:
    try:
        model, tok = FastLanguageModel.from_pretrained(load_from, max_seq_length=MAXLEN, dtype=torch.bfloat16, load_in_4bit=False)
    except TypeError as e:  # adapters saved by plain peft: load the base through unsloth, attach with peft (as rescore.py does)
        if "get_peft_model" not in str(e):
            raise
        from peft import PeftModel
        model, tok = FastLanguageModel.from_pretrained(model_id, max_seq_length=MAXLEN, dtype=torch.bfloat16, load_in_4bit=False)
        model = PeftModel.from_pretrained(model, load_from)
    if hasattr(model, "merge_and_unload"):
        model = model.merge_and_unload()
    FastLanguageModel.for_inference(model)
    cfg["merged"] = hasattr(model, "peft_config") is False
    tok.padding_side = "left"
    results = {}
    for cond, ctx in zip(conds, (False, True)):
        recs, mets = run_condition(model, tok, ctx)
        results[cond] = mets
        p = ROOT / "results" / "per_item" / f"gen_{stem}{sfx}.{cond}.jsonl"
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            for r in recs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        run.log(mets, condition=cond); run.artifact(p)
    OUT.write_text(json.dumps(results, indent=2)); run.artifact(OUT)

print(f"\n=== gen_eval {stem}{sfx} ===")
for cond, mets in results.items():
    print(f"-- {cond}")
    for k, v in mets.items():
        print(f"   {k:40s} {v}")
