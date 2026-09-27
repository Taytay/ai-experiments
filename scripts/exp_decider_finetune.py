"""Fine-tune a decider model (Mapika/decider-2b / -4b, Apache-2.0; PLAN step 77, MODEL-14) on the categoriser's own training episodes,
in decider's layout and readout: the episode's prompt without its "Category:" cue is the Context, the user's category names (after rename
augmentation) are the options, and the loss is cross-entropy over the option-label logits at the "Answer: (" slot (decider/train.py's
objective), options shuffled per episode by decider's own prompt.build. A rank-64 LoRA on every projection, the Gated DeltaNet ones
included, as row 78's Qwen3.5 categoriser; the same steps, batch and learning-rate schedule.

The episodes are exp_categoriser.py's sft_examples(), built under the same env (POI, FOLD, RENAME, POI_KIND, POI_DESC, POI_UNSEEN,
DBEP, ...), so the training users, layouts and augmentations match the categoriser arm it is compared with. ALL_LABELS has no analogue
here (one decision per episode).

env: MODEL (default Mapika/decider-2b), STEPS (800), MICRO (16 sequences per step), LR (1e-4), RUN_TAG, SEED, plus the episode knobs.
Writes models/adapters/decider_<model>_<episode suffix>_lora (peft) and results/decider_ft_<same>.json; score it with
exp_decision_models.py FAMILY=decider ADAPTER=<that dir name>. Tracker experiment "decider_finetune".
Row 79 (MODEL-16) uses it to train plain Qwen3.5 with one answer slot (MODEL=Qwen/Qwen3.5-2B; adapter slot_<model>_...).
usage: POI=poi1_v1 FOLD=0 RENAME=0.5 POI_KIND=1 POI_DESC=1 POI_UNSEEN=0.5 uv run --with transformers==5.17.0 --with flash-linear-attention
       --with "peft>=0.21" --with torch==2.13.0 --with torchvision==0.28.0 python scripts/exp_decider_finetune.py
"""
import importlib
import json
import os
import random
import sys
import time

from ai_experiments.evals.tracker import Run
from ai_experiments.paths import ROOT

MODEL = os.environ.get("MODEL", "Mapika/decider-2b")  # row 79: a plain Qwen3.5 (Qwen/Qwen3.5-2B) trained in decider's layout and readout
DECIDER_CODE = "Mapika/decider-2b"  # the repo whose decider/ package (prompt.build, the letter table) renders the episodes
STEPS, MICRO, LR = int(os.environ.get("STEPS", "800")), int(os.environ.get("MICRO", "16")), float(os.environ.get("LR", "1e-4"))
SEED = int(os.environ.get("SEED", "0"))
QUESTION = "Which of this user's categories does the last transaction belong to?"  # as exp_decision_models.py

# exp_categoriser.py's episode builder under the same env (its module level up to the model code: config, data, sft_examples)
os.environ.setdefault("LLM_BASE", "Qwen/Qwen3.5-2B")  # only for its name bookkeeping; no model is loaded from it here
sys.argv = [sys.argv[0], "llm", "none"]
_src = (ROOT / "scripts" / "exp_categoriser.py").read_text().split("\nTARGETS = [")[0]
C = {"__name__": "exp_categoriser_episodes", "__file__": str(ROOT / "scripts" / "exp_categoriser.py")}
exec(compile(_src, "exp_categoriser.py", "exec"), C)
SFX = C["SFX"].replace("_alllab", "").replace("_hf", "")
NAME = f"{'decider_' if 'decider' in MODEL else 'slot_'}{MODEL.split('/')[-1]}_{SFX}_lora"  # slot_: row 79's one-slot Qwen3.5
OUT_DIR = ROOT / "models" / "adapters" / NAME


def names_of(prompt):
    """The category names from the episode's header (plain "Categories: a, b" or the row 73 description list "- a: kinds")."""
    head = prompt.split("\n\n", 1)[0]
    if head.startswith("Categories:\n"):
        return [ln[2:].split(": ", 1)[0] for ln in head.split("\n")[1:]]
    return head[len("Categories: "):].split(", ")


def episodes():
    out = []
    for e in C["sft_examples"]():
        prompt, ans = e[0], e[1].strip()
        assert prompt.endswith("Category:")
        opts = names_of(prompt)
        out.append((prompt[: -len("Category:")].rstrip(), opts, opts.index(ans)))
    return out


def main():
    import torch
    import torch.nn.functional as F
    from huggingface_hub import snapshot_download
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer
    path = snapshot_download(MODEL)
    sys.path.insert(0, snapshot_download(DECIDER_CODE, allow_patterns=["decider/*"]))  # decider's prompt code; the weights may be any Qwen3.5 (row 79)
    P = importlib.import_module("decider.prompt")
    tok = AutoTokenizer.from_pretrained(path)
    lm = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16).cuda()
    lm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False}); lm.enable_input_require_grads()
    torch.manual_seed(SEED)
    targets = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj", "in_proj_qkv", "in_proj_z", "out_proj"]
    model = get_peft_model(lm, LoraConfig(r=64, lora_alpha=128, lora_dropout=0.0, bias="none", task_type="CAUSAL_LM", target_modules=targets))
    letters = torch.tensor(P.letter_ids(tok), device="cuda")
    head_w = lm.get_output_embeddings().weight  # decider ties or not; its readout is these rows (decider/model.py)

    class Q:
        def __init__(self, text, options, gold):
            self.text, self.options, self.gold = text, options, gold

    class Ex:
        def __init__(self, context, qs):
            self.context, self.qs = context, qs

    eps = episodes(); rng = random.Random(SEED)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=LR, weight_decay=0.0, betas=(0.9, 0.95))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 20) * max(0.0, 1 - s / STEPS))
    pad = tok.pad_token_id or 0
    model.train(); t0 = time.time(); losses = []; n_tok = 0
    print(f"   {len(eps)} episodes, {STEPS} steps x {MICRO}, lr {LR}, {sum(p.numel() for p in params)} trainable", flush=True)
    for step in range(STEPS):
        built = [P.build(Ex(s, [Q(QUESTION, o, g)]), tok, rng=rng, max_options=255, max_ctx_tokens=16384) for s, o, g in (rng.choice(eps) for _ in range(MICRO))]
        T = -(-max(len(b["ids"]) for b in built) // 64) * 64; n_tok += sum(len(b["ids"]) for b in built)
        ids = torch.full((len(built), T), pad, dtype=torch.long)
        att = torch.zeros_like(ids)
        for i, b in enumerate(built):
            ids[i, :len(b["ids"])] = torch.tensor(b["ids"]); att[i, :len(b["ids"])] = 1
        h = model.base_model.model.model(input_ids=ids.cuda(), attention_mask=att.cuda()).last_hidden_state
        hs = torch.stack([h[i, b["slots"][0]] for i, b in enumerate(built)])
        z = F.linear(hs, head_w[letters]).float()
        n = torch.tensor([b["nopts"][0] for b in built], device="cuda")
        z = z.masked_fill(torch.arange(z.shape[1], device="cuda")[None] >= n[:, None], float("-inf"))
        loss = F.cross_entropy(z, torch.tensor([b["golds"][0] for b in built], device="cuda"))
        loss.backward(); torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
        losses.append(loss.item())
        if step % 50 == 0 or step == STEPS - 1:
            print(f"   step {step} loss {sum(losses[-50:]) / len(losses[-50:]):.3f} ({(time.time() - t0) / 60:.1f} min)", flush=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True); model.save_pretrained(OUT_DIR)
    return dict(train_minutes=round((time.time() - t0) / 60, 1), tokens=n_tok, final_loss=round(sum(losses[-50:]) / len(losses[-50:]), 3),
                peak_alloc_GiB=round(torch.cuda.max_memory_allocated() / 2 ** 30, 2), adapter=str(OUT_DIR.relative_to(ROOT)), n_episodes=len(eps))


if __name__ == "__main__":
    from ai_experiments.licences import open_licence
    open_licence(MODEL)
    cfg = dict(model=MODEL, steps=STEPS, micro=MICRO, lr=LR, seed=SEED, episodes_sfx=SFX, poi=C["POI"], fold=C["FOLD"], rename=C["RENAME"],
               poi_kind=C["POI_KIND"], poi_desc=C["POI_DESC"], poi_unseen=C["POI_UNSEEN"], dbep=C["DBEP"], lora_r=64, question=QUESTION)
    with Run("decider_finetune", model=MODEL, config=cfg) as run:
        stats = main()
        print("===", stats, flush=True)
        run.log({k: v for k, v in stats.items() if isinstance(v, (int, float))})
        (ROOT / "results" / f"decider_ft_{NAME}.json").write_text(json.dumps(dict(cfg, **stats), indent=2))
