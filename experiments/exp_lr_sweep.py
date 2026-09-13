"""Follow-up: full FT at lr 5e-5 over-trained (ppl 16 -> 800). Repeat raw vs
augmented at lower learning rates, same 420 steps, and report WiSE-FT (alpha=0.5)
for each. Reuses definitions from exp_knowledge_injection.py."""
import json, sys, time
from pathlib import Path
import torch
here = Path(__file__).parent
src = (here / "exp_knowledge_injection.py").read_text().split("# ----------------------------------------------------------------------------- main")[0]
ns = {"__file__": str(here / "exp_knowledge_injection.py")}
exec(compile(src, "ki_defs", "exec"), ns)
load_base, train, evaluate, merchants, M = ns["load_base"], ns["train"], ns["evaluate"], ns["merchants"], ns["M"]
STEPS = ns["STEPS"]
raw_texts = [M.raw_fact(m) for m in merchants]
aug_texts = [t for m in merchants for t in M.augmented(m)]
OUT = here.parent / "results" / "lr_sweep.json"
results = {}
for lr in (1e-5, 2e-5):
    for name, texts in (("ft_raw", raw_texts), ("ft_aug", aug_texts)):
        key = f"{name}_lr{lr:g}"
        print(f"\n== {key}", flush=True)
        t0 = time.time()
        tok, model = load_base()
        base_sd = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        train(model, tok, texts, STEPS, lr)
        results[key] = evaluate(model, tok)
        with torch.no_grad():
            for k, v in model.state_dict().items():
                v.copy_(0.5 * v + 0.5 * base_sd[k].to(v.device, v.dtype))
        results[key + "_wise0.5"] = evaluate(model, tok)
        results[key]["minutes"] = round((time.time() - t0) / 60, 1)
        print("  ", results[key]); print("   wise:", results[key + "_wise0.5"], flush=True)
        del model; torch.cuda.empty_cache()
        OUT.write_text(json.dumps(results, indent=2))
cols = ["clean_category", "bank_category", "sells", "reverse", "ppl_general"]
print("\n=== SUMMARY ===")
print(f"{'condition':22s}" + "".join(f"{c:>16s}" for c in cols))
for k, r in results.items():
    print(f"{k:22s}" + "".join(f"{str(r.get(c, '')):>16s}" for c in cols))
