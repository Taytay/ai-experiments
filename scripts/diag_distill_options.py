"""Where the option-renormalised teacher is right and wrong (PLAN step 27, REPORT.md 18).

usage: uv run python scripts/diag_distill_options.py [adapter=models/adapters/curriculum_Qwen2.5-3B_P2_lora]

For every fact question of arm P2 (scripts/exp_distill_v2.py, five question forms per trained species),
scores the option set (a) with the teacher (base model, entry in context), (b) with the base model on the
bare prompt and (c) with the P2 adapter on the bare prompt, and reports per question form: accuracy of the
argmax over options, mean probability of the top option, mean probability of the gold option. Also the
same for the ladder's L1_recall / L1_recall_fmt items (the evaluation's questions) with context.
Writes results/distill_options_Qwen2.5-3B.json.
"""
import json
import sys

from ai_experiments.paths import ROOT

import unsloth  # noqa: F401
import torch
import torch.nn.functional as F
from unsloth import FastLanguageModel
from peft import PeftModel

from ai_experiments import items as I
from ai_experiments import universe as U
from ai_experiments.scoring import option_logprobs_batched

sys.argv = [sys.argv[0]] + sys.argv[1:]
ADAPTER = ROOT / (sys.argv[1] if len(sys.argv) > 1 else "models/adapters/curriculum_Qwen2.5-3B_P2_lora")
MAXLEN = 768
model, tok = FastLanguageModel.from_pretrained("Qwen/Qwen2.5-3B", max_seq_length=MAXLEN, dtype=torch.bfloat16, load_in_4bit=False)
model = PeftModel.from_pretrained(model, str(ADAPTER), adapter_name="P2")
FastLanguageModel.for_inference(model); model.eval()
species = U.build(); by_name = {s["name"]: s for s in species}; names_sorted = sorted(by_name, key=len, reverse=True)
pad = tok.pad_token_id or 0

Q_FORMS = [
    ("Question: What type is {N}?\nAnswer:", " {N} is a {V}-type.", "type", U.TYPE_LIST),
    ("Question: What is {N} weak to?\nAnswer:", " {N}, being {T}-type, is weak to {V}.", "weakness", U.TYPE_LIST),
    ("Question: Where does {N} live?\nAnswer:", " {N} lives in {V} habitats in {R}.", "habitat", U.HABITATS),
    ("Question: Where does {N} live?\nAnswer:", " {N} lives in {H} habitats in {V}.", "region", U.REGIONS),
    ("Question: What does {N} eat?\nAnswer:", " {N} is an {V}.", "diet", U.DIETS),
]


def guide_for(text):
    found = sorted(n for n in names_sorted if n in text)
    return "Field guide:\n" + "\n".join(U.entry(by_name[n]) for n in found) + "\n\n"


@torch.no_grad()
def option_lp(prompt, options, adapter):
    """The training-time scorer (scoring.option_logprobs_batched), one example."""
    ids_of = lambda t: tok(t, add_special_tokens=False)["input_ids"]
    p = ids_of(prompt); opts = [ids_of(o) for o in options]
    if adapter:
        model.set_adapter(adapter); return option_logprobs_batched(model, tok, [p], [opts])[0]
    with model.disable_adapter():
        return option_logprobs_batched(model, tok, [p], [opts])[0]


@torch.no_grad()
def option_lp_unpadded(prompt, options, adapter):
    """Reference: each option scored alone, no padding, full logits (the Scorer's arithmetic)."""
    ids_of = lambda t: tok(t, add_special_tokens=False)["input_ids"]
    p = ids_of(prompt); out = []
    for o in options:
        oi = ids_of(o); ids = torch.tensor([p + oi], device="cuda")
        if adapter:
            model.set_adapter(adapter); lg = model(input_ids=ids).logits.float()
        else:
            with model.disable_adapter():
                lg = model(input_ids=ids).logits.float()
        out.append(F.log_softmax(lg[0, len(p) - 1:-1], -1).gather(-1, ids[0, len(p):].unsqueeze(-1)).sum())
    return torch.stack(out)


@torch.no_grad()
def option_lp_mixed_batch(items, adapter):
    """The training-time situation: several examples of very different lengths in one batch."""
    ids_of = lambda t: tok(t, add_special_tokens=False)["input_ids"]
    ps = [ids_of(it["prompt"]) for it in items]; os_ = [[ids_of(o) for o in it["options"]] for it in items]
    if adapter:
        model.set_adapter(adapter); return option_logprobs_batched(model, tok, ps, os_)
    with model.disable_adapter():
        return option_logprobs_batched(model, tok, ps, os_)


def run(items, label):
    agg = {}
    for it in items:
        for cond, (prompt, adapter) in {"teacher": (guide_for(it["prompt"]) + it["prompt"], None), "base": (it["prompt"], None),
                                        "P2": (it["prompt"], "P2")}.items():
            q = F.softmax(option_lp(prompt, it["options"], adapter), -1)
            d = agg.setdefault((it["form"], cond), dict(n=0, acc=0.0, top_p=0.0, gold_p=0.0))
            d["n"] += 1; d["acc"] += float(q.argmax().item() == it["answer"]); d["top_p"] += q.max().item(); d["gold_p"] += q[it["answer"]].item()
    print(f"\n{label}\n{'form':16s} {'cond':8s} {'n':>4s} {'acc':>6s} {'top_p':>6s} {'gold_p':>6s}")
    out = {}
    for (form, cond), d in sorted(agg.items()):
        n = d["n"]; row = dict(n=n, acc=round(100 * d["acc"] / n, 1), top_p=round(d["top_p"] / n, 3), gold_p=round(d["gold_p"] / n, 3))
        out[f"{form}/{cond}"] = row
        print(f"{form:16s} {cond:8s} {n:4d} {row['acc']:6.1f} {row['top_p']:6.3f} {row['gold_p']:6.3f}")
    return out


@torch.no_grad()
def option_lp_unpadded(prompt, options, adapter):
    """Reference: each option scored alone, no padding (the Scorer's arithmetic)."""
    ids_of = lambda t: tok(t, add_special_tokens=False)["input_ids"]
    p = ids_of(prompt); out = []
    for o in options:
        oi = ids_of(o); ids = torch.tensor([p + oi], device="cuda")
        if adapter:
            model.set_adapter(adapter); lg = model(input_ids=ids).logits.float()
        else:
            with model.disable_adapter():
                lg = model(input_ids=ids).logits.float()
        lp = F.log_softmax(lg[0, len(p) - 1:-1], -1).gather(-1, ids[0, len(p):].unsqueeze(-1)).sum()
        out.append(lp)
    return torch.stack(out)


facts = []
for s in species:
    if s["heldout"]:
        continue
    f = dict(N=s["name"], T=s["type"], H=s["habitat"], R=s["region"])
    for q, opt, attr, vals in Q_FORMS:
        facts.append(dict(prompt=q.format(**f), options=[opt.format(V=v, **f) for v in vals], answer=list(vals).index(s[attr]), form=attr))
# padding cross-check: the batched scorer, alone and inside a mixed-length batch, against one-option-at-a-time scoring
worst, worst_mixed = 0.0, 0.0
probe = facts[::40][:17]
for it in probe:
    for adapter in (None, "P2"):
        for prompt in (it["prompt"], guide_for(it["prompt"]) + it["prompt"]):
            a = option_lp(prompt, it["options"], adapter); b = option_lp_unpadded(prompt, it["options"], adapter)
            worst = max(worst, (a - b).abs().max().item())
episodes = U.episodes(species, n=8, seed=3, ctx_frac=0.0)
long_items = [dict(prompt=guide_for(e["prompt"]) + e["prompt"], options=[" " + l for l in e["labels"]]) for e in episodes]
for it in probe[:8]:
    ctx_prompt = guide_for(it["prompt"]) + it["prompt"]
    batch = long_items + [dict(prompt=ctx_prompt, options=it["options"])]
    mixed = option_lp_mixed_batch(batch, None)[-1]; ref = option_lp_unpadded(ctx_prompt, it["options"], None)
    worst_mixed = max(worst_mixed, (mixed - ref).abs().max().item())
print(f"\npadding cross-check: max |batched - unpadded| = {worst:.4f} nats (68 single-example pairs); "
      f"inside a batch with eight 400-token episode rows = {worst_mixed:.4f} nats (8 fact questions)")
frozen = I.load_all(morph=False)
ladder = [dict(it, form=it["level"]) for it in frozen.ladder if it["level"] in ("L1_recall", "L1_recall_fmt")]
res = dict(padding_check_max_abs_diff=round(worst, 4), padding_check_mixed_batch_max_abs_diff=round(worst_mixed, 4), training_questions=run(facts, "P2 training questions (680)"), ladder_L1=run(ladder, "ladder L1 items (320)"))
(ROOT / "results" / "distill_options_Qwen2.5-3B.json").write_text(json.dumps(res, indent=1))
print("wrote results/distill_options_Qwen2.5-3B.json")
