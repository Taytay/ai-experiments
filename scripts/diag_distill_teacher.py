"""How sharp is the prompt-distillation teacher? (REPORT.md section 14, BASE-3, PLAN step 7)

usage: uv run python scripts/diag_distill_teacher.py [out.json]

For each trained species, the declarative training sentence "{N} is a {T}-type creature. ..." and
the probability of the type's first token at its position, for: the base model on the bare
sentence; the teacher of scripts/exp_distill.py (base model with the species' field-guide entry
prepended); the same at temperature 2; and the P and A adapters on the bare sentence. Also the
cloze prompt "Question: What type is {N}?\\nAnswer:" for base, P and A. Prints mean probability,
tempered probability, argmax accuracy and the scaled KL to the tempered teacher; writes the
per-species numbers to out.json (default results/distill_teacher_Qwen2.5-3B.json). About 3 minutes.
"""
import json, math, sys
import torch, torch.nn.functional as F
from unsloth import FastLanguageModel
from peft import PeftModel
from ai_experiments import universe as U
from ai_experiments.paths import ROOT

MAXLEN, TEMP = 768, 2.0
model, tok = FastLanguageModel.from_pretrained("Qwen/Qwen2.5-3B", max_seq_length=MAXLEN, dtype=torch.bfloat16, load_in_4bit=False)
model = PeftModel.from_pretrained(model, str(ROOT / "models/adapters/curriculum_Qwen2.5-3B_P_lora"), adapter_name="P")
model.load_adapter(str(ROOT / "models/adapters/curriculum_Qwen2.5-3B_A_lora"), adapter_name="A")
FastLanguageModel.for_inference(model); model.eval()
species = [s for s in U.build() if not s["heldout"]]


@torch.no_grad()
def dist(text, target_char, ctx="", adapter=None):
    """log-probs (T=1 and tempered) at the position predicting the token that covers target_char."""
    full = ctx + text
    enc = tok(full, return_offsets_mapping=True, add_special_tokens=False, return_tensors="pt")
    offs = enc.pop("offset_mapping")[0].tolist()
    pos = next(i for i, (a, b) in enumerate(offs) if a <= len(ctx) + target_char < b)
    ids = enc["input_ids"].cuda()
    if adapter is None:
        with model.disable_adapter():
            logits = model(input_ids=ids).logits[0, pos - 1].float()
    else:
        model.set_adapter(adapter)
        logits = model(input_ids=ids).logits[0, pos - 1].float()
    return ids[0, pos].item(), F.log_softmax(logits, -1), F.log_softmax(logits / TEMP, -1)


rows = []
for s in species:
    text = U._DESC[0].format(N=s["name"], T=s["type"], W=s["weakness"], H=s["habitat"], D=s["diet"], R=s["region"], S=s["stage"])
    tchar = text.index(f" {s['type']}-type") + 1
    ctx = U.entry(s) + "\n\n"
    qa = f"Question: What type is {s['name']}?\nAnswer:"
    qa_text = qa + f" {s['type']}"
    r = {}
    for name, kw in (("base", {}), ("teacher", dict(ctx=ctx)), ("P", dict(adapter="P")), ("A", dict(adapter="A"))):
        tid, lp, lpT = dist(text, tchar, **kw)
        r[name] = dict(p=math.exp(lp[tid].item()), pT=math.exp(lpT[tid].item()), top=lp.argmax().item() == tid, lpT=lpT)
    for name, kw in (("base_qa", {}), ("P_qa", dict(adapter="P")), ("A_qa", dict(adapter="A"))):
        tid, lp, lpT = dist(qa_text, len(qa) + 1, **kw)
        r[name] = dict(p=math.exp(lp[tid].item()), pT=math.exp(lpT[tid].item()), top=lp.argmax().item() == tid, lpT=lpT)
    t = r["teacher"]["lpT"]
    for name in ("base", "P", "A"):
        r[name]["kl_T"] = F.kl_div(r[name]["lpT"], t, log_target=True, reduction="sum").item() * TEMP ** 2
    rows.append(r)

names = ["base", "teacher", "P", "A", "base_qa", "P_qa", "A_qa"]
print(f"{'model':10s} {'mean p(type)':>13s} {'mean p_T2':>10s} {'argmax acc':>11s} {'KL_T2*T2 to teacher':>20s}   (n={len(rows)})")
for n in names:
    p = sum(r[n]["p"] for r in rows) / len(rows); pT = sum(r[n]["pT"] for r in rows) / len(rows)
    acc = 100 * sum(r[n]["top"] for r in rows) / len(rows)
    kl = sum(r[n]["kl_T"] for r in rows) / len(rows) if "kl_T" in rows[0][n] else float("nan")
    print(f"{n:10s} {p:13.3f} {pT:10.3f} {acc:11.1f} {kl:20.3f}")
json.dump([{k: {kk: vv for kk, vv in v.items() if kk != "lpT"} for k, v in r.items()} for r in rows],
          open(sys.argv[1] if len(sys.argv) > 1 else ROOT / "results" / "distill_teacher_Qwen2.5-3B.json", "w"), indent=1)
