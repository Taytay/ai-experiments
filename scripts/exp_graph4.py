"""Score the GRAPH-4 items (PLAN step 30; `universe.graph4_items`, frozen as data/processed/graph4_v1{,_wind}.json) on saved weights:
the induced type -> weakness edge, asked bare ("What type are T-type creatures weak to?", eight items) and in the two-hop path form
(the species' type stated, its weakness asked, 136 items). Every `exp_curriculum.py` run from this commit scores them itself;
this back-fills the saved adapters.

usage: uv run python scripts/exp_graph4.py base | <adapter dir under models/adapters>      WEAKNESS=independent for the _wind set / adapters
outputs: results/graph4_<tag>.json, results/per_item/graph4_<tag>.<cond>.jsonl; tracker experiment "graph4"
"""
import json
import os
import sys
import time

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import unsloth  # noqa: F401
import torch
from unsloth import FastLanguageModel

from ai_experiments import items as I
from ai_experiments.evals.tracker import Run
from ai_experiments.paths import ROOT
from ai_experiments.scoring import Scorer, aggregate, per_item_path, write_records

WHAT = sys.argv[1] if len(sys.argv) > 1 else "base"
MODEL = os.environ.get("MODEL", "Qwen/Qwen2.5-3B")
LOAD_4BIT = bool(int(os.environ.get("LOAD_4BIT", "0")))  # unsloth's loader defaults to the NF4 4-bit base (load_in_4bit=True); until 2026-09-21 this script
# never set it, so its saved results were read on the 4-bit base (REPORT.md section 44). Default now bf16, like exp_curriculum; LOAD_4BIT=1 reproduces the old reads.
WEAKNESS = os.environ.get("WEAKNESS", "type")
tag = ("base" if WHAT == "base" else WHAT) + ("_wind" if WEAKNESS == "independent" else "")
tag += ("_" + os.environ["RUN_TAG"]) if os.environ.get("RUN_TAG") else ""  # e.g. RUN_TAG=bf16 for the section 44 re-reads beside the 4-bit files
OUT = ROOT / "results" / f"graph4_{tag}.json"
doc = I.load(I.GRAPH4, False, weakness=WEAKNESS)
items = doc["items"]
src = MODEL if WHAT == "base" else str(ROOT / "models" / "adapters" / WHAT)
cond = "base" if WHAT == "base" else "trained"

with Run("graph4", model=MODEL, config=dict(what=WHAT, weakness=WEAKNESS, graph4_sha=doc["sha256"], n_items=len(items))) as run:
    model, tok = FastLanguageModel.from_pretrained(src, max_seq_length=768, dtype=torch.bfloat16, load_in_4bit=LOAD_4BIT)
    tok.padding_side = "right"; model.eval(); t0 = time.time()
    sc = Scorer(model, tok, maxlen=768, extras=False)
    recs = sc.score(items, label="graph4")
    m = aggregate(recs)
    # the bare level has eight items: report the count of types answered right, and the path level by type
    m["G4_bare_correct_of_8"] = int(sum(r["correct"] for r in recs if r["level"] == "G4_type_weakness_bare"))
    by_type = {}
    for r, it in zip(recs, items):
        if it["level"] == "G4_type_weakness_path":
            by_type.setdefault(it["type"], []).append(r["correct"])
    m["G4_path_types_above_half"] = int(sum(sum(v) / len(v) > 0.5 for v in by_type.values()))
    m["eval_minutes"] = round((time.time() - t0) / 60, 1)
    results = {cond: m}
    OUT.parent.mkdir(exist_ok=True); OUT.write_text(json.dumps(results, indent=2))
    run.log(m, condition=cond); run.artifact(OUT)
    run.artifact(write_records(per_item_path(OUT, cond), recs))
print(f"=== graph4 {tag}: {m}")
