"""Check the Scorer's prompt-cache path against the packed path (2026-09-26): the same items scored both ways, per-option sum log-prob
differences, prediction agreement and time. Loads the model as exp_real6.py does (unsloth, or SCORER=hf for transformers + peft).
usage: [SCORER=hf] [N=300] [EXTRAS=1] uv run python scripts/check_scorer_cache.py <adapter dir under models/adapters | base model id> <item set>
"""
import json
import os
import sys
import time

import torch

from ai_experiments.paths import PROCESSED, ROOT
from ai_experiments.scoring import Scorer

what, items_set = sys.argv[1], sys.argv[2]
N, SCORER, EXTRAS = int(os.environ.get("N", "300")), os.environ.get("SCORER", "unsloth"), bool(int(os.environ.get("EXTRAS", "0")))
items = json.loads((PROCESSED / f"{items_set}.json").read_text())["items"][:N]
adapter = ROOT / "models" / "adapters" / what
src = str(adapter) if adapter.exists() else what
if SCORER == "hf":
    from transformers import AutoModelForCausalLM, AutoTokenizer
    base = json.loads((adapter / "adapter_config.json").read_text())["base_model_name_or_path"] if adapter.exists() else what
    tok = AutoTokenizer.from_pretrained(base)
    model = AutoModelForCausalLM.from_pretrained(base, dtype=torch.bfloat16, attn_implementation="sdpa").cuda()
    if adapter.exists():
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, src)
else:
    import unsloth  # noqa: F401
    from unsloth import FastLanguageModel
    model, tok = FastLanguageModel.from_pretrained(src, max_seq_length=2048, dtype=torch.bfloat16, load_in_4bit=False)
tok.padding_side = "right"; model.eval()
out = {}
for cache in (False, True, False):  # packed twice: the packed path's own run-to-run noise is the yardstick
    sc = Scorer(model, tok, maxlen=2048, extras=EXTRAS, rows_per_forward=16, tokens_per_forward=24576, prefix_cache=cache)
    torch.cuda.synchronize(); t0 = time.time()
    recs = sc.score(items)
    torch.cuda.synchronize(); dt = time.time() - t0
    out.setdefault(cache, []).append((recs, dt))
    print(f"prefix_cache={cache}: {len(items)} items in {dt:.1f} s", flush=True)
(p1, t1), (p2, t2) = out[False]
(c1, tc) = out[True][0]


def cmp(a, b, key="sum_lp"):
    d = [abs(x - y) for ra, rb in zip(a, b) for x, y in zip(ra[key], rb[key])]
    agree = sum(ra["pred"] == rb["pred"] for ra, rb in zip(a, b)) / len(a)
    return max(d), sum(d) / len(d), agree


for name, (a, b) in {"packed vs packed": (p1, p2), "packed vs cache": (p1, c1)}.items():
    mx, mean, agree = cmp(a, b)
    print(f"{name}: sum_lp |diff| max {mx:.4f} mean {mean:.5f}; same prediction {100 * agree:.1f}%")
    if EXTRAS:
        for k in ("dc_lp", "unc_lp", "mcf_lp", "hyb_lp"):
            print(f"   {k}: |diff| max {cmp(a, b, k)[0]:.4f}")
print(f"accuracy packed {100 * sum(r['correct'] for r in p1) / len(p1):.1f}, cache {100 * sum(r['correct'] for r in c1) / len(c1):.1f}; "
      f"speed-up {min(t1, t2) / tc:.1f}x")
