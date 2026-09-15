"""Corpus perplexity for the base model and every saved adapter of one base (PLAN step 24, TRAIN-3).

usage: uv run python scripts/corpus_ppl.py [Qwen/Qwen2.5-3B] [--adapters NAME ...] [--note TEXT]

REPORT.md 15.5: the one-paragraph general-text perplexity (`merchants.GENERAL_TEXT`, 249 words) swung
2x between checkpoints of one run and between two runs of one recipe. This scores the frozen
WikiText-2 slice (`data/processed/corpus_ppl_v1.json`, 29 paragraphs, about 5,000 tokens,
`ai_experiments.items`) beside that paragraph, for the base model and each LoRA adapter under
models/adapters/ whose `base_model_name_or_path` is the given model (default: all of them). The base
is loaded once and the adapters are attached and switched with peft, so each adapter costs seconds
rather than a model load. Nothing else is re-scored; from this commit on every evaluation in
exp_curriculum.py, exp_distill.py and rescore.py reports the same number as `L7_ppl_wikitext`.

Outputs: results/corpus_ppl_<tag>.json (per model: ppl, mean NLL, its standard error over paragraphs,
the paired NLL difference to the base with its standard error, per-paragraph NLLs, the paragraph ppl),
reports/corpus_ppl_<tag>.md (the table). Tracker: experiment "corpus_ppl", one run per base model,
condition = adapter folder name without "_lora" ("base" for the bare model).
"""
import argparse
import json
import math
import time

from ai_experiments.paths import ADAPTERS, ROOT

import unsloth  # noqa: F401  (before transformers)
import torch
from unsloth import FastLanguageModel

from ai_experiments import items as I
from ai_experiments.evals.tracker import Run
from ai_experiments.merchants import GENERAL_TEXT
from ai_experiments.scoring import perplexity

MAXLEN = 768

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("base", nargs="?", default="Qwen/Qwen2.5-3B")
ap.add_argument("--adapters", nargs="*", help="adapter folder names under models/adapters/ (default: every adapter of this base)")
ap.add_argument("--note")
a = ap.parse_args()
tag = a.base.split("/")[-1]


def adapters_of(base_tag):
    out = []
    for d in sorted(ADAPTERS.iterdir()):
        cfg = d / "adapter_config.json"
        if cfg.exists() and json.loads(cfg.read_text()).get("base_model_name_or_path", "").split("/")[-1] == base_tag:
            out.append(d.name)
    return out


names = a.adapters if a.adapters is not None else adapters_of(tag)
FROZEN = I.load_all(morph=False)
if not FROZEN.corpus:
    raise SystemExit("data/processed/corpus_ppl_v1.json is missing: uv run python -m ai_experiments.items freeze-corpus")
print(f"corpus perplexity | base {a.base} | {len(FROZEN.corpus)} paragraphs | adapters: {', '.join(names) or 'none'}", flush=True)


@torch.no_grad()
def paragraph_nlls(model, tok):
    """(sum NLL, token count) per paragraph of the frozen slice."""
    torch.cuda.empty_cache()
    out = []
    for c in FROZEN.corpus:
        ids = tok(c, return_tensors="pt", add_special_tokens=False)["input_ids"][:, :MAXLEN].cuda()
        logits = model(input_ids=ids).logits.float()
        out.append((torch.nn.functional.cross_entropy(logits[0, :-1], ids[0, 1:], reduction="sum").item(), ids.shape[1] - 1))
    return out


def summarise(rows, base_rows=None):
    nll_sum, n = sum(r[0] for r in rows), sum(r[1] for r in rows)
    per = [s / k for s, k in rows]
    mean = nll_sum / n
    sd = lambda xs: (sum((x - sum(xs) / len(xs)) ** 2 for x in xs) / (len(xs) - 1)) ** 0.5
    m = dict(ppl=round(math.exp(mean), 3), nll=round(mean, 4), nll_se=round(sd(per) / len(per) ** 0.5, 4), n_tokens=n,
             n_paragraphs=len(rows), per_paragraph_nll=[round(x, 4) for x in per])
    if base_rows is not None:
        d = [s / k - bs / bk for (s, k), (bs, bk) in zip(rows, base_rows)]
        m["nll_diff_vs_base"] = round(sum(d) / len(d), 4)
        m["nll_diff_se"] = round(sd(d) / len(d) ** 0.5, 4)
    return m


cfg = dict(base_model=a.base, adapters=names, maxlen=MAXLEN, n_paragraphs=len(FROZEN.corpus), **FROZEN.config())
results = {}
with Run("corpus_ppl", model=a.base, config=cfg, note=a.note) as run:
    model, tok = FastLanguageModel.from_pretrained(a.base, max_seq_length=MAXLEN, dtype=torch.bfloat16, load_in_4bit=False)
    FastLanguageModel.for_inference(model); model.eval()
    t0 = time.time()
    base_rows = paragraph_nlls(model, tok)
    results["base"] = dict(**summarise(base_rows), L7_ppl_general=round(perplexity(model, tok, GENERAL_TEXT), 2))
    print(f"    base: wikitext ppl {results['base']['ppl']} (nll {results['base']['nll']} +- {results['base']['nll_se']}), "
          f"paragraph ppl {results['base']['L7_ppl_general']}  [{time.time() - t0:.0f}s]", flush=True)
    if names:
        from peft import PeftModel
        key = lambda n: n.replace(".", "_")  # peft adapter names are module names: no dots
        model = PeftModel.from_pretrained(model, str(ADAPTERS / names[0]), adapter_name=key(names[0]))
        for n in names[1:]:
            model.load_adapter(str(ADAPTERS / n), adapter_name=key(n))
        model.eval()
        for n in names:
            model.set_adapter(key(n))
            cond = n.removesuffix("_lora")
            rows = paragraph_nlls(model, tok)
            results[cond] = dict(**summarise(rows, base_rows), L7_ppl_general=round(perplexity(model, tok, GENERAL_TEXT), 2))
            r = results[cond]
            print(f"    {cond}: wikitext ppl {r['ppl']} (nll {r['nll']} +- {r['nll_se']}; vs base {r['nll_diff_vs_base']:+.4f} "
                  f"+- {r['nll_diff_se']}), paragraph ppl {r['L7_ppl_general']}  [{time.time() - t0:.0f}s]", flush=True)
    for cond, m in results.items():
        run.log({"L7_ppl_wikitext": m["ppl"], "L7_nll_wikitext": m["nll"], "L7_nll_wikitext_se": m["nll_se"],
                 "L7_ppl_general": m["L7_ppl_general"], **({"L7_nll_wikitext_diff_vs_base": m["nll_diff_vs_base"],
                                                             "L7_nll_wikitext_diff_se": m["nll_diff_se"]} if "nll_diff_vs_base" in m else {})},
                condition=cond)
    out = ROOT / "results" / f"corpus_ppl_{tag}.json"
    out.write_text(json.dumps(dict(base=a.base, maxlen=MAXLEN, corpus_sha256=FROZEN.sha.get(I.CORPUS), results=results), indent=1))
    run.artifact(out)
    md = [f"# Corpus perplexity: {tag}", "",
          f"Generated by `scripts/corpus_ppl.py` (PLAN step 24). WikiText-2 test slice `data/processed/corpus_ppl_v1.json` "
          f"({len(FROZEN.corpus)} paragraphs, {results['base']['n_tokens']} scored tokens), token-weighted; the standard error is over "
          f"paragraph means; the difference column is the paired per-paragraph NLL difference to the base model with its standard "
          f"error. The last column is the old one-paragraph number (`merchants.GENERAL_TEXT`).", "",
          "| model | WikiText ppl | mean NLL +- se | NLL - base (paired) +- se | paragraph ppl |", "|---|---|---|---|---|"]
    for cond, m in results.items():
        diff = f"{m['nll_diff_vs_base']:+.3f} +- {m['nll_diff_se']:.3f}" if "nll_diff_vs_base" in m else ""
        md.append(f"| {cond} | {m['ppl']:.2f} | {m['nll']:.3f} +- {m['nll_se']:.3f} | {diff} | {m['L7_ppl_general']:.2f} |")
    rep = ROOT / "reports" / f"corpus_ppl_{tag}.md"
    rep.write_text("\n".join(md) + "\n", encoding="utf-8")
    run.artifact(rep)
    print(f"wrote {out.relative_to(ROOT)} and {rep.relative_to(ROOT)}")
