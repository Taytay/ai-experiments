"""Score any saved adapter (or a bare model) on the frozen item sets and keep the per-item scores.

usage: uv run python scripts/rescore.py MODEL_OR_ADAPTER [--tag TAG] [--morph] [--note TEXT]

  MODEL_OR_ADAPTER  a hub id (Qwen/Qwen2.5-3B) or an adapter folder (models/adapters/<name>);
                    an adapter folder loads its base model from adapter_config.json
  --tag             name of the output files (default: adapter folder name without "_lora",
                    or the model id after "/")
  --morph           score the morphology universe's item sets (arms E / base_m) instead of the plain ones

Curriculum arms re-score themselves with `EVAL_ONLY=1 scripts/exp_curriculum.py ARM`; this script
covers everything else, e.g. the adapters from scripts/exp_universe_ladder.py, which predate the
tracker's EVAL_ONLY path. Same scorer, same items, same record format (ai_experiments.scoring):

  results/rescore_<tag>.json                          metrics: {"trained": ..., "trained_ctx": ...}
                                                      ("base"/"base_ctx" for a bare model)
  results/per_item/rescore_<tag>.<condition>.jsonl    per item, per option

Recorded in the tracker as experiment "rescore" with the item-set hashes in the config.
SMOKE=1 (plumbing check) scores 1/40 of the items, writes *_smoke files and records nothing.
"""
import argparse
import json
import os
import time
from pathlib import Path

from ai_experiments.paths import ROOT

import unsloth  # noqa: F401  (before transformers)
import torch
from unsloth import FastLanguageModel

from ai_experiments import items as I
from ai_experiments.evals.tracker import Run
from ai_experiments.merchants import GENERAL_TEXT
from ai_experiments.scoring import Scorer, aggregate, per_item_path, perplexity, write_records

MAXLEN = 768  # same as exp_curriculum.py

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("target")
ap.add_argument("--tag")
ap.add_argument("--morph", action="store_true")
ap.add_argument("--note")
a = ap.parse_args()

target = Path(a.target)
is_adapter = target.is_dir() and (target / "adapter_config.json").exists()
if target.exists() and not is_adapter:
    raise SystemExit(f"{target} exists but has no adapter_config.json; pass a LoRA adapter folder or a hub model id")
tag = a.tag or (target.name.removesuffix("_lora") if is_adapter else a.target.split("/")[-1])
SMOKE = bool(os.environ.get("SMOKE"))
OUT = ROOT / "results" / f"rescore_{tag}{'_smoke' if SMOKE else ''}.json"
cond = "trained" if is_adapter else "base"
base = json.loads((target / "adapter_config.json").read_text())["base_model_name_or_path"] if is_adapter else a.target

FROZEN = I.load_all(morph=a.morph)
ladder, probes, suite = FROZEN.ladder, FROZEN.probes, FROZEN.suite
if SMOKE:
    ladder, probes, suite = ladder[::40], probes[::12], suite[::48]
print(f"rescore {a.target} -> {OUT.name} | base {base} | items {FROZEN.version} ({'morph' if a.morph else 'plain'}): "
      f"{len(ladder)} ladder | {len(probes)} probes | {len(suite)} ICL suite", flush=True)

cfg = dict(target=str(target.relative_to(ROOT)) if is_adapter and target.is_absolute() else a.target, base_model=base,
           tag=tag, is_adapter=is_adapter, maxlen=MAXLEN, **FROZEN.config())
with Run("rescore", model=base, config=cfg, note=a.note, enabled=not SMOKE) as run:
    model, tok = FastLanguageModel.from_pretrained(str(target) if is_adapter else a.target, max_seq_length=MAXLEN,
                                                   dtype=torch.bfloat16, load_in_4bit=False)
    tok.padding_side = "right"
    model.eval(); torch.cuda.empty_cache(); t0 = time.time()
    sc = Scorer(model, tok, maxlen=MAXLEN)
    recs = {cond: sc.score(ladder, label="ladder") + sc.score(probes, label="probe") + sc.score(suite, label="ICL suite"),
            cond + "_ctx": sc.score(ladder, ctx=True, label="ladder+ctx")}
    results = {c: aggregate(r) for c, r in recs.items()}
    m = results[cond]
    sym = [v for k, v in m.items() if k.startswith("ICL_symbol")]
    nat = [v for k, v in m.items() if k.startswith("ICL_natural")]
    m["ICL_symbol_mean"], m["ICL_natural_mean"] = round(sum(sym) / len(sym), 1), round(sum(nat) / len(nat), 1)
    m["L7_ppl_general"] = round(perplexity(model, tok, GENERAL_TEXT), 2)
    m["eval_minutes"] = round((time.time() - t0) / 60, 1)
    OUT.write_text(json.dumps(results, indent=2))
    run.artifact(OUT)
    for c, r in recs.items():
        run.log(results[c], condition=c)
        run.artifact(write_records(per_item_path(OUT, c), r))

print(f"\n=== rescore {tag} ===")
for c, mets in results.items():
    print(f"-- {c}")
    for k, v in mets.items():
        print(f"   {k:32s} {v}")
