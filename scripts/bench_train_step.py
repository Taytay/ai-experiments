"""Where a training step's time goes (row 78: Qwen3.5-2B trained at ~3.7k tokens/s against Qwen2.5-3B's ~14k through unsloth).
One LoRA (r64, as exp_categoriser.py) on the model, BATCH (8) real POI-1 items of ~900 tokens per step, timed after warm-up under:
  full     the training loop as exp_categoriser.py's TRAINER=hf path: logits at every position, float32 cross-entropy
  labelled hidden states only, the LM head applied at the labelled positions (~1,200 of ~14,000)
  each with and without gradient checkpointing; plus which gated-delta / conv1d implementation the model's layers hold.
usage: MODEL=Qwen/Qwen3.5-2B uv run [overlay] python scripts/bench_train_step.py
"""
import json
import os
import random
import time

if os.environ.get("BENCH_UNSLOTH"):  # import unsloth first, as exp_categoriser's unsloth path does: its zoo patches in the vendored fla kernels
    import unsloth  # noqa: F401
import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from ai_experiments.paths import PROCESSED

MODEL = os.environ.get("MODEL", "Qwen/Qwen3.5-2B")
LABEL = os.environ.get("LABEL", MODEL)  # printed on every line: jobs in one launch interleave their logs
BATCH = int(os.environ.get("BATCH", "8"))
N_STEPS = int(os.environ.get("N_STEPS", "6"))


def batches(tok, n=BATCH):
    doc = json.loads((PROCESSED / "poi1_v1_kinds.json").read_text())
    rng = random.Random(0); out = []
    for _ in range(N_STEPS + 2):
        its = rng.sample(doc["items"], n)
        seqs = [tok(it["prompt"] + it["options"][it["answer"]], add_special_tokens=False)["input_ids"] for it in its]
        L = max(map(len, seqs))
        ids = torch.tensor([s + [tok.pad_token_id or 0] * (L - len(s)) for s in seqs])
        lab = torch.full_like(ids, -100)
        for i, s in enumerate(seqs):  # ~12% of positions labelled, as ALL_LABELS episodes
            for p in rng.sample(range(1, len(s)), len(s) // 12):
                lab[i, p] = s[p]
        att = (torch.arange(L)[None] < torch.tensor([len(s) for s in seqs])[:, None]).long()
        out.append((ids.cuda(), lab.cuda(), att.cuda()))
    return out


def main():
    tok = AutoTokenizer.from_pretrained(MODEL)
    lm = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, attn_implementation="sdpa").cuda()
    targets = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"] + (["in_proj_qkv", "in_proj_z", "out_proj"] if "Qwen3.5" in MODEL else [])
    model = get_peft_model(lm, LoraConfig(r=64, lora_alpha=128, lora_dropout=0.0, bias="none", task_type="CAUSAL_LM", target_modules=targets))
    for m in lm.modules():
        if hasattr(m, "chunk_gated_delta_rule"):
            print(LABEL, "gated-delta:", m.chunk_gated_delta_rule.__module__, m.chunk_gated_delta_rule.__name__, "| conv1d fn:", getattr(m.causal_conv1d_fn, "__module__", None), flush=True)
            break
    bs = batches(tok); inner = model.get_base_model()
    n_tok = sum(int(a.sum()) for _, _, a in bs[2:])
    res = {}
    for ckpt in (True, False):
        if ckpt:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False}); model.enable_input_require_grads()
        else:
            model.gradient_checkpointing_disable()
        for mode in ("labelled", "full"):
            torch.cuda.reset_peak_memory_stats(); times = []
            for k, (ids, lab, att) in enumerate(bs):
                torch.cuda.synchronize(); t0 = time.time()
                if mode == "full":
                    logits = model(input_ids=ids, attention_mask=att).logits[:, :-1]  # per sequence, as exp_categoriser.py
                    n_lab = max(int((lab[:, 1:] != -100).sum()), 1)
                    loss = sum(F.cross_entropy(logits[i].float(), lab[i, 1:], ignore_index=-100, reduction="sum") for i in range(len(ids))) / n_lab
                else:
                    h = inner.model(input_ids=ids, attention_mask=att).last_hidden_state[:, :-1]
                    tgt = lab[:, 1:]; sel = tgt != -100
                    loss = F.cross_entropy(inner.lm_head(h[sel]).float(), tgt[sel])
                loss.backward(); model.zero_grad(set_to_none=True)
                torch.cuda.synchronize()
                if k >= 2:
                    times.append(time.time() - t0)
            s = sum(times)
            res[f"{mode}{'_ckpt' if ckpt else ''}"] = dict(sec_per_step=round(s / len(times), 3), tok_per_s=round(n_tok / s), peak_GiB=round(torch.cuda.max_memory_allocated() / 2 ** 30, 1))
            print(LABEL, mode, "ckpt" if ckpt else "no-ckpt", res[f"{mode}{'_ckpt' if ckpt else ''}"], flush=True)
    print("===", LABEL, json.dumps(dict(model=MODEL, batch=BATCH, **res)), flush=True)


if __name__ == "__main__":
    main()
