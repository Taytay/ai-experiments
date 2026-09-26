"""Score the PLAN step 20 item sets on saved weights (EVAL-4, EVAL-5): the identifiable induction items (I2_*, without and with
the field-guide entries in context), the out-of-distribution ICL suite (ICL2_*) and the 5-shot MMLU slice (K_mmlu), plus the
original suite and ARC-Easy for the paired comparison. No training; every future `exp_curriculum.py` run scores these sets
itself, this script back-fills the saved adapters.

usage: uv run python scripts/exp_items_v2.py base                       the untrained base (Qwen/Qwen2.5-3B, or MODEL=...)
       uv run python scripts/exp_items_v2.py <adapter dir under models/adapters/>
       uv run python scripts/exp_items_v2.py full <dir of a full checkpoint>   (an edited model of step 19)
env: SMOKE=1 (every 20th item, no tracker), MODEL (base model id, default Qwen/Qwen2.5-3B)
outputs: results/items2_<tag>.json (tag = base | adapter name | checkpoint basename), per-item files under results/per_item/,
tracker experiment "items_v2"
"""
import json
import os
import sys
import time

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True,garbage_collection_threshold:0.8")

import unsloth  # noqa: F401
import torch
from unsloth import FastLanguageModel

from ai_experiments import items as I
from ai_experiments.evals.tracker import Run
from ai_experiments.paths import ROOT
from ai_experiments.scoring import Scorer, aggregate, per_item_path, write_records

MODEL = os.environ.get("MODEL", "Qwen/Qwen2.5-3B")
SMOKE = bool(os.environ.get("SMOKE"))
MAXLEN = 1536  # MMLU 5-shot prompts run to about 900 tokens; the scorer cuts prompts to MAXLEN - 32 and options must fit in the rest
what = sys.argv[1] if len(sys.argv) > 1 else "base"
if what == "base":
    src, tag, kind = MODEL, "base" if MODEL == "Qwen/Qwen2.5-3B" else f"base_{MODEL.split('/')[-1]}", "base"  # another base model gets its own file
elif what == "full":
    src, tag, kind = sys.argv[2], os.path.basename(sys.argv[2].rstrip("/")), "full"
else:
    src, tag, kind = str(ROOT / "models" / "adapters" / what), what, "adapter"
OUT = ROOT / "results" / f"items2_{tag}{'_smoke' if SMOKE else ''}.json"

F = I.load_all(morph=False)
assert F.induction2 and F.suite2 and F.mmlu, "freeze the v2 sets first: uv run python -m ai_experiments.items freeze-v2"
ind, suite2, mmlu, suite, known = F.induction2, F.suite2, F.mmlu, F.suite, F.known
if SMOKE:
    ind, suite2, mmlu, suite, known = ind[::20], suite2[::20], mmlu[::20], suite[::20], known[::20]

cfg = dict(source=src, kind=kind, tag=tag, maxlen=MAXLEN, **F.config())
with Run("items_v2", model=MODEL, config=cfg, enabled=not SMOKE) as run:
    model, tok = FastLanguageModel.from_pretrained(src, max_seq_length=MAXLEN, dtype=torch.bfloat16)
    tok.padding_side = "right"
    model.eval(); t0 = time.time()
    sc = Scorer(model, tok, maxlen=MAXLEN, extras=False)
    recs = {"noctx": sc.score(ind, label="induction2") + sc.score(suite2, label="ICL suite2") + sc.score(mmlu, label="MMLU")
            + sc.score(suite, label="ICL suite") + sc.score(known, label="ARC-Easy"),
            "ctx": sc.score(ind, ctx=True, label="induction2+ctx")}
    m = aggregate(recs["noctx"])
    for pre in ("ICL_symbol", "ICL_natural", "ICL2_symbol", "ICL2_natural"):
        vs = [v for k, v in m.items() if k.startswith(pre + "_")]
        if vs:
            m[pre + "_mean"] = round(sum(vs) / len(vs), 1)
    m["eval_minutes"] = round((time.time() - t0) / 60, 1)
    results = {"trained" if kind != "base" else "base": m, ("trained" if kind != "base" else "base") + "_ctx": aggregate(recs["ctx"])}
    OUT.parent.mkdir(exist_ok=True); OUT.write_text(json.dumps(results, indent=2))
    for cond, mets in results.items():
        run.log(mets, condition=cond)
    run.artifact(OUT)
    for key, cond in (("noctx", "trained" if kind != "base" else "base"), ("ctx", ("trained" if kind != "base" else "base") + "_ctx")):
        run.artifact(write_records(per_item_path(OUT, cond), recs[key]))
    print(f"=== items2 {tag}")
    for cond, mets in results.items():
        print(f"-- {cond}")
        for k, v in mets.items():
            print(f"   {k:28s} {v}")
