"""PLAN step 151 (MODEL-22; owner 2026-10-01: use strands' own training recipe): the categoriser's recipe episodes as strands-decider
training data, for its own trainer (`python -m strands_decider.cli train --config`). The episodes are the ones exp_strands_finetune.py
draws (exp_decider_finetune's episodes(), abstain_aug, evfree_aug under the same env): STEPS x MICRO draws with the same seed, so both
routes see the same rows; strands' collator shuffles the options. Written to OUT/:
  train.jsonl    strands Examples: kind choice, the episode context as state, exp_decision_models.QUESTION, the category names (no
                 descriptions), the gold index. An evidence-free episode (the recipe's soft target) gets weight 0, so it carries no label
                 loss, and a uniform teacher row
  teacher.jsonl  {"i", "probs"} for the evidence-free episodes (uniform) and those where the user's choice is random (the true split)
  train.yaml     strands' v19 reference config (configs/train.yaml: Qwen3.5-2B-Base, pointer head 256, rank-16 LoRA on v19's targets,
                 KL 0.3 to the frozen torso's option-number readout, lr 1e-4 / head 1e-3, batch 8 x 4, one epoch, warmup 3%), with these
                 files, teacher weight 1.0 (as v19's replay rows), max_length 4096 and output_dir models/adapters/strands_recipe_<suffix>
usage: <recipe env> uv run --with strands-decider==0.1.0 python scripts/export_strands_episodes.py OUT
"""
import json
import os
import random
import runpy
import sys
from pathlib import Path

from ai_experiments.paths import ROOT

out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
os.environ["INIT"] = "base"
sys.argv = [sys.argv[0]]
M = runpy.run_path(str(ROOT / "scripts" / "exp_strands_finetune.py"), run_name="export")
G = M["G"]
eps = G["episodes"](); rng = random.Random(G["SEED"])
name = f"strands_recipe_{G['SFX']}"
n = n_soft = 0
with open(out / "train.jsonl", "w") as f, open(out / "teacher.jsonl", "w") as ft:
    for _ in range(G["STEPS"] * G["MICRO"]):
        e = G["evfree_aug"](G["abstain_aug"](rng.choice(eps), rng), rng)
        ctx, opts, gold = e[0], list(e[1]), e[2]
        dist = G["SOFT_CTX"].get(ctx)
        f.write(json.dumps(dict(kind="choice", state=ctx, instructions=G["QUESTION"], options=[[o, ""] for o in opts],
                                label=0 if gold is None else gold, task="categoriser", weight=0.0 if gold is None else 1.0), ensure_ascii=False) + "\n")
        if gold is None:
            ft.write(json.dumps({"i": n, "probs": [1.0 / len(opts)] * len(opts)}) + "\n"); n_soft += 1
        elif dist:
            ft.write(json.dumps({"i": n, "probs": [dist.get(o, 0.0) for o in opts]}) + "\n"); n_soft += 1
        n += 1
cfg = dict(train_files=[str(out / "train.jsonl")], val_fraction=0.03, base_model="Qwen/Qwen3.5-2B-Base", num_slots=24, head_type="pointer",
           pointer_dim=256, head_init="random", kl_frozen_weight=0.3, teacher_file=str(out / "teacher.jsonl"), teacher_weight=1.0,
           head_dropout=0.05, max_length=4096, use_lora=True, lora_r=16, lora_alpha=32,
           lora_targets=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj", "in_proj_qkv", "in_proj_z", "in_proj_a",
                         "in_proj_b", "out_proj"],
           freeze_torso=False, epochs=1, micro_batch_size=8, grad_accum=4, lr=1e-4, head_lr=1e-3, weight_decay=0.01, warmup_ratio=0.03,
           max_grad_norm=1.0, gradient_checkpointing=True, shuffle_options=True, group_by_length=True, ordinal_smoothing=0.1,
           reverse_score_prob=0.5, output_dir=str(ROOT / "models" / "adapters" / name), seed=G["SEED"], log_every=20, eval_every=100)
import yaml  # noqa: E402
(out / "train.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
print(f"{n} rows ({n_soft} with a teacher distribution) -> {out}; checkpoint {name}")
