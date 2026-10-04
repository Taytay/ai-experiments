"""Ask decider readers what kind of business a bank string is (owner, 2026-10-03: "Can you ask the model to tell you what kind of
merchant it thinks that is?" about a missed first-time payee). One multiple-choice question per string, the options the 60 kinds of
ai_experiments.taxonomy_v2, decider's own layout (Context / Question / Options / "Answer: (", rand255 labels), one readout; prints the
top 5 kinds per reader. Readers: decider-4B untrained and the adapters named in ADAPTERS (comma list of models/adapters names).
env: STRINGS ("|"-separated bank strings), ADAPTERS.
usage: STRINGS="1-800 CONTACTS, INC.|SQ *RADIO COFFEE" ADAPTERS=<a>,<b> uv run --with transformers==5.17.0 ... python scripts/ask_kind.py
"""
import importlib
import math
import os
import random
import sys

from ai_experiments import oneslot
from ai_experiments import taxonomy_v2 as TX
from ai_experiments.paths import ROOT

STRINGS = [s for s in os.environ["STRINGS"].split("|") if s]
ADAPTERS = [a for a in os.environ.get("ADAPTERS", "").split(",") if a]
QUESTION = "What kind of business is this payee?"

if __name__ == "__main__":
    import torch
    import torch.nn.functional as F
    from huggingface_hub import snapshot_download
    from transformers import AutoModelForCausalLM, AutoTokenizer
    path = snapshot_download("Mapika/decider-4b")
    sys.path.insert(0, snapshot_download("Mapika/decider-2b", allow_patterns=["decider/*"]))
    P = importlib.import_module("decider.prompt")
    tok = AutoTokenizer.from_pretrained(path)
    kinds = [v for k, v in TX.KINDS.items() if k not in ("several", "purpose")]
    for reader in ["decider-4B, untrained"] + ADAPTERS:
        lm = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16).cuda().eval()
        if reader in ADAPTERS:
            from peft import PeftModel
            lm = PeftModel.from_pretrained(lm, str(ROOT / "models" / "adapters" / reader)).merge_and_unload().eval()
        print(f"\n=== {reader if reader not in ADAPTERS else reader[-60:]}", flush=True)
        for s in STRINGS:
            agg = [0.0] * len(kinds)
            for seed in range(4):  # average over four option orders and label draws (decider's option-order bias, REPORT 50)
                b = oneslot.build(P, tok, f"Payee as it appears on a bank statement: {s}", QUESTION, kinds, 0, random.Random(seed), labels="rand255")
                with torch.no_grad():
                    h = lm.model(input_ids=torch.tensor([b["ids"]]).cuda()).last_hidden_state[0, b["slot"]]
                    lp = F.log_softmax(F.linear(h.float(), lm.lm_head.weight[torch.tensor(b["labs"]).cuda()].float()), -1).tolist()
                for j, oi in enumerate(b["perm"]):
                    agg[oi] += math.exp(lp[j]) / 4
            top = sorted(range(len(kinds)), key=lambda k: -agg[k])[:5]
            print(f"  {s!r:38s} " + " | ".join(f"{kinds[k][:34]} {agg[k]:.0%}" for k in top), flush=True)
        del lm; torch.cuda.empty_cache()
