"""PLAN step 154 (MODEL-24, owner 2026-10-01: train decider's method on strands-decider's datasets and compare with strands). decider's
one-slot readout (ai_experiments.corpus_slot: decider's layout and labels, the label tokens' logits from the LM head) trained on strands'
v19 corpus (scripts/build_strands_corpus.sh), from the same torso as v19 (Qwen/Qwen3.5-2B-Base), with the parts of strands' recipe that
are about the data kept: its training files, option shuffling (scores in order or reversed), ordinal smoothing 0.1, row weights,
instruction variants, its replay distributions on the multi-step rows (KL, weight 1), one epoch, effective batch 32. What is decider's:
the readout, the layout and labels, and our LoRA (rank 64 on every projection, the DeltaNet ones included; lr 1e-4).

env: MODEL (Qwen/Qwen3.5-2B-Base), CORPUS (data/external/strands_corpus), MICRO (16), ACCUM (2), LR (1e-4), EPOCHS (1), SEED, MAX_STEPS
     (0 = the epoch), TEACH_W (1.0), RUN_TAG (h100bf16).
Writes models/adapters/corpusslot_<model>_<RUN_TAG><seed>_lora (peft) and results/corpusslot_ft_<same>.json.
Research only (owner, 2026-10-01): the corpus includes datasets outside our open-licence list; the adapter is never shipped.
usage: uv run --with transformers==5.17.0 --with flash-linear-attention --with "peft>=0.21" --with torch==2.13.0 --with torchvision==0.28.0
       python scripts/exp_corpus_slot.py
"""
import json
import os
import random
import sys
import time

from ai_experiments import corpus_slot as CS
from ai_experiments.evals.tracker import Run
from ai_experiments.paths import ROOT

MODEL = os.environ.get("MODEL", "Qwen/Qwen3.5-2B-Base")
CORPUS = ROOT / os.environ.get("CORPUS", "data/external/strands_corpus")
MICRO, ACCUM = int(os.environ.get("MICRO", "16")), int(os.environ.get("ACCUM", "2"))
LR, EPOCHS, SEED = float(os.environ.get("LR", "1e-4")), int(os.environ.get("EPOCHS", "1")), int(os.environ.get("SEED", "0"))
MAX_STEPS = int(os.environ.get("MAX_STEPS", "0"))
TEACH_W = float(os.environ.get("TEACH_W", "1.0"))
RUN_TAG = os.environ.get("RUN_TAG", "h100bf16")
FILES = ["train_v5", "multistep_v14", "generated_v16", "generated_v18", "adequacy_hs2", "adequacy_gen"]  # v19's train_files, in order
NAME = f"corpusslot_{MODEL.split('/')[-1].lower()}_{RUN_TAG}{'' if SEED == 0 else f's{SEED}'}_lora"
OUT_DIR = ROOT / "models" / "adapters" / NAME


def load_rows():
    rows = []
    for f in FILES:
        rows += [json.loads(l) for l in open(CORPUS / f"{f}.jsonl") if l.strip()]
    teach, bad = {}, 0
    for l in open(CORPUS / "replay_v14_multistep.jsonl"):  # strands' replay targets: {"i": row index in the concatenation, "probs"}
        d = json.loads(l)
        if d["i"] < len(rows) and len(d["probs"]) == len(rows[d["i"]]["options"]):
            teach[d["i"]] = d["probs"]
        else:
            bad += 1
    print(f"   {len(rows):,} rows from {len(FILES)} files; replay targets on {len(teach):,} ({bad} not matching)", flush=True)
    return rows, teach


def batches(rows, rng, lengths):
    """Length-grouped micro-batches (as strands' group_by_length): shuffle, cut into mega-batches of 50 micro-batches, sort each by
    length, split, shuffle the micro-batch order."""
    idx = list(range(len(rows))); rng.shuffle(idx)
    out = []
    mega = 50 * MICRO
    for k in range(0, len(idx), mega):
        chunk = sorted(idx[k:k + mega], key=lambda i: lengths[i])
        out += [chunk[j:j + MICRO] for j in range(0, len(chunk), MICRO)]
    rng.shuffle(out)
    return out


def main():
    import importlib
    import torch
    import torch.nn.functional as F
    from huggingface_hub import snapshot_download
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer
    sys.path.insert(0, snapshot_download("Mapika/decider-2b", allow_patterns=["decider/*"]))
    P = importlib.import_module("decider.prompt")
    tok = AutoTokenizer.from_pretrained(MODEL)
    lm = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).cuda()
    lm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False}); lm.enable_input_require_grads()
    torch.manual_seed(SEED)
    targets = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj", "in_proj_qkv", "in_proj_z", "out_proj"]
    model = get_peft_model(lm, LoraConfig(r=64, lora_alpha=128, lora_dropout=0.0, bias="none", task_type="CAUSAL_LM", target_modules=targets))
    head_w = lm.get_output_embeddings().weight
    rows, teach = load_rows()
    rng = random.Random(SEED)
    lengths = [len(CS.state_text(r["state"])) for r in rows]
    plan = [b for _ in range(EPOCHS) for b in batches(rows, rng, lengths)]
    steps = len(plan) // ACCUM if not MAX_STEPS else MAX_STEPS
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=LR, weight_decay=0.0, betas=(0.9, 0.95))
    warm = max(1, int(0.03 * steps))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / warm) * max(0.0, 1 - s / steps))
    pad = tok.pad_token_id or 0
    model.train(); t0 = time.time(); losses = []; n_tok = 0
    print(f"   {len(plan)} micro-batches of {MICRO}, accum {ACCUM}: {steps} steps, lr {LR}, warmup {warm}", flush=True)
    for step in range(steps):
        for a in range(ACCUM):
            mb = plan[step * ACCUM + a]
            built = [CS.render(P, tok, rows[i], rng, train=True) for i in mb]
            T = -(-max(len(b["ids"]) for b in built) // 64) * 64; n_tok += sum(len(b["ids"]) for b in built)
            ids = torch.full((len(built), T), pad, dtype=torch.long); att = torch.zeros_like(ids)
            for i, b in enumerate(built):
                ids[i, :len(b["ids"])] = torch.tensor(b["ids"]); att[i, :len(b["ids"])] = 1
            h = model.base_model.model.model(input_ids=ids.cuda(), attention_mask=att.cuda()).last_hidden_state
            per, w = [], []
            for i, (ri, b) in enumerate(zip(mb, built)):
                lp = F.log_softmax(F.linear(h[i, b["slot"]], head_w[torch.tensor(b["labs"], device="cuda")]).float(), -1)
                t = torch.tensor(CS.target(rows[ri], b["order"]), device="cuda")
                loss_i = -(t * lp).sum()
                if ri in teach and TEACH_W:  # KL(teacher || student) on the options as shown
                    q = torch.tensor([teach[ri][k] for k in b["order"]], device="cuda").clamp_min(1e-8); q = q / q.sum()
                    loss_i = loss_i + TEACH_W * (q * (q.log() - lp)).sum()
                per.append(loss_i); w.append(float(rows[ri].get("weight", 1.0)))
            wt = torch.tensor(w, device="cuda")
            loss = (torch.stack(per) * wt).sum() / wt.sum().clamp_min(1e-8) / ACCUM
            loss.backward(); losses.append(loss.item() * ACCUM)
        torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
        if step % 100 == 0 or step == steps - 1:
            print(f"   step {step}/{steps} loss {sum(losses[-200:]) / len(losses[-200:]):.3f} ({(time.time() - t0) / 60:.1f} min)", flush=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True); model.save_pretrained(OUT_DIR)
    return dict(train_minutes=round((time.time() - t0) / 60, 1), tokens=n_tok, steps=steps, rows=len(rows),
                final_loss=round(sum(losses[-200:]) / len(losses[-200:]), 3), peak_alloc_GiB=round(torch.cuda.max_memory_allocated() / 2 ** 30, 2),
                adapter=str(OUT_DIR.relative_to(ROOT)))


if __name__ == "__main__":
    from ai_experiments.licences import open_licence
    open_licence(MODEL)
    cfg = dict(model=MODEL, corpus=str(CORPUS.relative_to(ROOT)), files=FILES, micro=MICRO, accum=ACCUM, lr=LR, epochs=EPOCHS, seed=SEED,
               teach_w=TEACH_W, lora_r=64, labels="decider letters", max_steps=MAX_STEPS)
    with Run("corpus_slot", model=MODEL, config=cfg) as run:
        stats = main()
        print("===", stats, flush=True)
        run.log({k: v for k, v in stats.items() if isinstance(v, (int, float))})
        (ROOT / "results" / f"corpusslot_ft_{NAME}.json").write_text(json.dumps(dict(cfg, **stats), indent=2))
