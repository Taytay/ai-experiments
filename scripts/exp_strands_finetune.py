"""PLAN step 151 (MODEL-22, owner 2026-10-01): strands-decider trained on the categoriser's episodes. The episodes, their sampling and
augmentations are exp_decider_finetune.py's (its module level up to main: episodes(), abstain_aug, evfree_aug, the soft targets, under
the same env as the decider-4B recipe it is compared with: DBEP, FOLD, RENAME, MISLEAD, ALT, LOOKUP, OVERRIDE, EVFREE, EMPTY, OTHERS, ...);
the model, prompt and readout are strands' (ai_experiments.strands): options shuffled per episode, cross-entropy over the pointer head's
option logits (a uniform target on evidence-free episodes, the true split where the user's choice is random, as in the recipe). No
shot-label token loss (the recipe's AUX_LM; strands' torso has no LM head in use) and no KL to the frozen torso (KL_W, not implemented).

  INIT=v19   continue the published checkpoint (StrandsAgents/strands-decider-2B-hobson-v19): its LoRA and pointer head trained further
  INIT=base  strands' architecture from Qwen3.5-2B-Base: a fresh rank-16 LoRA (v19's targets) and a fresh pointer head (dim 256)

env: INIT, STEPS (800), MICRO (16), LR (1e-4, LoRA), HEAD_LR (1e-3 for a fresh head, 2e-4 for v19's), SEED, RUN_TAG, + the episode knobs.
Writes models/adapters/strands_<init>_<episode suffix> (strands' checkpoint layout: lora/, slot_head.pt, config,
tokenizer) and results/strands_ft_<same>.json. Score with exp_decision_models.py FAMILY=strands MODEL=<V19> ADAPTER=<that name>.
usage: INIT=v19 OTHERS=0.5 ... uv run --with strands-decider==0.1.0 --with transformers==5.17.0 --with flash-linear-attention
       --with "peft>=0.21" --with torch==2.13.0 --with torchvision==0.28.0 python scripts/exp_strands_finetune.py
"""
import json
import os
import random
import sys
import time

from ai_experiments import strands as SD
from ai_experiments.evals.tracker import Run
from ai_experiments.paths import ROOT

INIT = os.environ.get("INIT", "v19")
assert INIT in ("v19", "base"), INIT
os.environ["MODEL"] = SD.V19  # only names exp_decider_finetune's (unused) run; the episodes do not depend on it
LABELLED = os.environ.get("STATE", "") == "labelled"  # row 152 (MODEL-23): our labelled layout inside strands' state (SD.labelled)
AUX_SHOTS = float(os.environ.get("AUX_SHOTS", "0"))  # row 152: + w x the loss on the history rows' codes (the recipe's shot-label loss)
BASE = os.environ.get("BASE", SD.BASE)  # row 153: Qwen/Qwen3.5-4B-Base for a 4B torso (INIT=base only)
POINTER_DIM = int(os.environ.get("POINTER_DIM", "256"))  # row 153: scaled with the torso's width (320 at 4B: 256 x 2560 / 2048)
HEAD_LR = float(os.environ.get("HEAD_LR", "2e-4" if INIT == "v19" else "1e-3"))
_argv = sys.argv
_src = (ROOT / "scripts" / "exp_decider_finetune.py").read_text().split("\ndef main():")[0]
G = {"__name__": "decider_episodes", "__file__": str(ROOT / "scripts" / "exp_decider_finetune.py")}
exec(compile(_src, "exp_decider_finetune.py", "exec"), G)
sys.argv = _argv
STEPS, MICRO, LR, SEED = G["STEPS"], G["MICRO"], G["LR"], G["SEED"]
assert not G["AUX_LM"] and not G["TEACHER"] and not G["LAYOUT"], "AUX_LM, TEACHER and LAYOUT are the decider readout's; not used here"
assert INIT == "base" or (BASE == SD.BASE and POINTER_DIM == 256), "BASE / POINTER_DIM apply to INIT=base"
NAME = f"strands_{INIT}{'' if BASE == SD.BASE else '-' + BASE.split('-')[-2].lower()}{'' if POINTER_DIM == 256 else f'_pd{POINTER_DIM}'}_{G['SFX']}" + ("_lab" if LABELLED else "") + (f"_aux{round(AUX_SHOTS * 100)}" if AUX_SHOTS else "")
assert not AUX_SHOTS or LABELLED, "AUX_SHOTS needs STATE=labelled (the codes it predicts)"  # the episode suffix carries RUN_TAG and the seed
OUT_DIR = ROOT / "models" / "adapters" / NAME


def build_model():
    import torch
    from strands_decider.modeling import StrandsDeciderConfig, StrandsDeciderModel
    if INIT == "v19":
        m = SD.load(SD.V19, device="cpu")
        for n, p in m.torso.named_parameters():
            p.requires_grad_("lora_" in n)
        m.config.temperature, m.config.temperature_by_kind = 1.0, {}  # v19's fitted temperatures describe it, not the fine-tune
    else:
        from strands_decider.modeling import checkpoint_dir, config_path
        v19 = StrandsDeciderConfig.from_json(config_path(checkpoint_dir(SD.V19)))  # v19's architecture: pointer dim, LoRA rank and targets
        cfg = StrandsDeciderConfig(**{**v19.__dict__, "temperature": 1.0, "temperature_by_kind": {}, "kl_frozen_weight": 0.0,
                                      "base_model": BASE, "pointer_dim": POINTER_DIM})
        torch.manual_seed(SEED)
        m = StrandsDeciderModel.from_pretrained_base(cfg)
        if m.tokenizer.pad_token is None:
            m.tokenizer.pad_token = m.tokenizer.eos_token
    m.config.max_length = 16384  # nothing is truncated (encode never cuts); the saved window says so
    for p in m.head.parameters():
        p.requires_grad_(True)
    base = getattr(m.torso, "base_model", m.torso)
    inner = getattr(base, "model", base)
    inner.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    return m.cuda()


def main():
    import torch
    m = build_model()
    eps = G["episodes"](); rng = random.Random(SEED)
    lora = [p for n, p in m.torso.named_parameters() if p.requires_grad]
    head = list(m.head.parameters())
    opt = torch.optim.AdamW([{"params": lora, "lr": LR}, {"params": head, "lr": HEAD_LR}], weight_decay=0.01, betas=(0.9, 0.95))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 20) * max(0.0, 1 - s / STEPS))
    m.train(); t0 = time.time(); losses = []; n_tok = 0; longest = 0
    print(f"   {NAME}: {len(eps)} episodes, {STEPS} steps x {MICRO}, lr {LR} / head {HEAD_LR}, "
          f"{sum(p.numel() for p in lora + head):,} trainable", flush=True)
    for step in range(STEPS):
        picked = [G["evfree_aug"](G["abstain_aug"](rng.choice(eps), rng), rng) for _ in range(MICRO)]
        encs, targets = [], []
        for e in picked:
            ctx, opts, gold = e[0], list(e[1]), e[2]
            perm = rng.sample(range(len(opts)), len(opts))
            if LABELLED:
                st, shown, spans = SD.labelled(ctx, opts, rng)
                encs.append(SD.encode_aux(m.tokenizer, st, G["QUESTION"], shown, spans, order=perm))
            else:
                encs.append(SD.encode(m.tokenizer, ctx, G["QUESTION"], opts, order=perm))
            dist = G["SOFT_CTX"].get(ctx)
            if gold is None:  # evidence-free (soft): uniform over the options
                t = [1.0 / len(opts)] * len(opts)
            elif dist:  # the user's choice is random here: the true split
                t = [dist.get(opts[k], 0.0) for k in perm]
            else:
                t = [float(k == gold) for k in perm]
            targets.append(t)
        longest = max([longest] + [len(x[0]) for x in encs]); n_tok += sum(len(x[0]) for x in encs)
        lp, hid = SD.forward(m, encs)
        tv = torch.zeros_like(lp)
        for i, t in enumerate(targets):
            tv[i, :len(t)] = torch.tensor(t, device=lp.device)
        loss = -(tv * lp.masked_fill(torch.isinf(lp), 0.0)).sum(-1).mean()
        if AUX_SHOTS:  # the codes in the history rows, each from the position before, through the tied embeddings
            rows_ = [(i, t, y) for i, e in enumerate(encs) for t, y in e[2]]
            if rows_:
                hs = hid[torch.tensor([r[0] for r in rows_], device="cuda"), torch.tensor([r[1] for r in rows_], device="cuda")]
                lg = (hs @ m._output_embedding().t()).float()
                loss = loss + AUX_SHOTS * torch.nn.functional.cross_entropy(lg, torch.tensor([r[2] for r in rows_], device="cuda"))
        loss.backward(); torch.nn.utils.clip_grad_norm_(lora + head, 1.0); opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
        losses.append(loss.item())
        if step % 50 == 0 or step == STEPS - 1:
            print(f"   step {step} loss {sum(losses[-50:]) / len(losses[-50:]):.3f} longest {longest} ({(time.time() - t0) / 60:.1f} min)", flush=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True); m.save_pretrained(str(OUT_DIR))
    if LABELLED:  # row 152: how to read this checkpoint (exp_decision_models picks it up)
        (OUT_DIR / "strands_extra.json").write_text(json.dumps(dict(state="labelled")))
    return dict(train_minutes=round((time.time() - t0) / 60, 1), tokens=n_tok, longest=longest, final_loss=round(sum(losses[-50:]) / len(losses[-50:]), 3),
                peak_alloc_GiB=round(torch.cuda.max_memory_allocated() / 2 ** 30, 2), adapter=str(OUT_DIR.relative_to(ROOT)), n_episodes=len(eps))


if __name__ == "__main__":
    from ai_experiments.licences import open_licence
    open_licence(SD.V19); open_licence(BASE)
    C = G["C"]
    cfg = dict(model=SD.V19 if INIT == "v19" else BASE, pointer_dim=POINTER_DIM, init=INIT, state="labelled" if LABELLED else "plain", aux_shots=AUX_SHOTS, steps=STEPS, micro=MICRO, lr=LR, head_lr=HEAD_LR, seed=SEED, episodes_sfx=G["SFX"],
               fold=C["FOLD"], rename=C["RENAME"], dbep=C["DBEP"], lora_r=16, question=G["QUESTION"], evfree=G["EVFREE"], evfree_mode=G["EVFREE_MODE"])
    with Run("strands_finetune", model=cfg["model"], config=cfg) as run:
        stats = main()
        print("===", stats, flush=True)
        run.log({k: v for k, v in stats.items() if isinstance(v, (int, float))})
        (ROOT / "results" / f"strands_ft_{NAME}.json").write_text(json.dumps(dict(cfg, **stats), indent=2))
