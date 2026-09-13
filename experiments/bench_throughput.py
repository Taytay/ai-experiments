"""Training / inference throughput and memory benchmark on this GPU.

For each (model, method, seq_len) config: warm up, time N optimizer steps on
synthetic token batches (isolates GPU from tokenization), record tokens/s,
step time, peak VRAM, and model-FLOPs utilization (MFU) against the GPU's dense
bf16 peak. Also measures forward-only (eval/scoring) throughput and the MiniLM
contrastive loop. Optionally tries unsloth's FastLanguageModel LoRA path.

usage: uv run python experiments/bench_throughput.py [quick]
Results -> results/bench_throughput.json and the evals tracker (experiment "bench_throughput").
"""
import gc
import json
import sys
import time
import traceback
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
from evals.tracker import Run  # noqa: E402

QUICK = len(sys.argv) > 1 and sys.argv[1] == "quick"
OUT = Path(__file__).parent.parent / "results" / "bench_throughput.json"
GPU = torch.cuda.get_device_name(0)
# dense bf16 tensor-core peak (FP32 accumulate), no sparsity
PEAK_TFLOPS = {"RTX 3090": 71.0, "RTX 4090": 165.0, "A100": 312.0, "H100": 989.0, "RTX 5090": 210.0}
peak = next((v for k, v in PEAK_TFLOPS.items() if k in GPU), 71.0)
N_STEPS = 6 if QUICK else 12
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def n_params(model):
    return sum(p.numel() for p in model.parameters())


def cleanup():
    gc.collect(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()


def load(model_id, method):
    from transformers import AutoModelForCausalLM
    kw = {}
    if method == "qlora_4bit":
        from transformers import BitsAndBytesConfig
        kw["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                                       bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
        kw["dtype"] = torch.bfloat16
    else:
        kw["dtype"] = torch.float32 if method == "full_fp32master" else torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(model_id, **kw)
    if method != "qlora_4bit":
        model = model.cuda()
    return model


def prepare(model, method, grad_ckpt):
    if method.startswith("lora") or method == "qlora_4bit":
        from peft import LoraConfig, get_peft_model
        if method == "qlora_4bit":
            from peft import prepare_model_for_kbit_training
            model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=grad_ckpt)
        elif grad_ckpt:
            model.gradient_checkpointing_enable(); model.enable_input_require_grads()
        model = get_peft_model(model, LoraConfig(r=64, lora_alpha=128, lora_dropout=0.0, target_modules=LORA_TARGETS))
    elif grad_ckpt:
        model.gradient_checkpointing_enable()
    return model


def bench_train(model_id, method, seq_len, bs, grad_ckpt=False, unsloth=False):
    name = f"{model_id.split('/')[-1]}|{'unsloth_' if unsloth else ''}{method}{'+gc' if grad_ckpt else ''}|L{seq_len}|bs{bs}"
    print(f"\n== {name}", flush=True)
    cleanup(); t_load = time.time()
    try:
        if unsloth:
            from unsloth import FastLanguageModel
            model, _ = FastLanguageModel.from_pretrained(model_id, max_seq_length=seq_len, dtype=torch.bfloat16,
                                                         load_in_4bit=(method == "qlora_4bit"))
            model = FastLanguageModel.get_peft_model(model, r=64, lora_alpha=128, lora_dropout=0.0, target_modules=LORA_TARGETS,
                                                     use_gradient_checkpointing="unsloth" if grad_ckpt else False)
            base_params = sum(p.numel() for n, p in model.named_parameters() if "lora" not in n)
        else:
            model = load(model_id, method)
            base_params = n_params(model)
            model = prepare(model, method, grad_ckpt)
        load_s = time.time() - t_load
        params = [p for p in model.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(params, lr=1e-5)
        vocab = model.config.vocab_size if hasattr(model, "config") else 150000
        model.train()
        autocast = method == "full_fp32master"

        def step():
            ids = torch.randint(0, vocab, (bs, seq_len), device="cuda")
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=autocast):
                loss = model(input_ids=ids, labels=ids).loss
            loss.backward(); opt.step(); opt.zero_grad(set_to_none=True)

        for _ in range(3):
            step()
        torch.cuda.synchronize(); t0 = time.time()
        for _ in range(N_STEPS):
            step()
        torch.cuda.synchronize(); dt = (time.time() - t0) / N_STEPS
        toks = bs * seq_len / dt
        # FLOPs/token: full FT 6N; LoRA ~4N (frozen weight grads skipped); grad ckpt adds ~2N recompute
        fpt = (6 if method == "full_fp32master" else 4) * base_params + (2 * base_params if grad_ckpt else 0)
        res = dict(ok=1, base_params_B=round(base_params / 1e9, 3), trainable_M=round(sum(p.numel() for p in params) / 1e6, 1),
                   step_ms=round(dt * 1000, 1), tokens_per_s=round(toks), samples_per_s=round(bs / dt, 2),
                   peak_alloc_GiB=round(torch.cuda.max_memory_allocated() / 2**30, 2),
                   peak_reserved_GiB=round(torch.cuda.max_memory_reserved() / 2**30, 2),
                   mfu_pct=round(100 * toks * fpt / (peak * 1e12), 1), load_s=round(load_s, 1))
    except Exception as e:  # OOM or unsupported path: record and move on
        res = dict(ok=0, error=f"{type(e).__name__}: {str(e)[:200]}")
        traceback.print_exc(limit=1)
    print("  ", res, flush=True)
    for v in ("model", "opt", "params"):
        if v in locals():
            del locals()[v]
    cleanup()
    return name, res


def bench_forward(model_id, seq_len, bs):
    """Eval / likelihood-scoring throughput (no grad), bf16."""
    name = f"{model_id.split('/')[-1]}|forward_bf16|L{seq_len}|bs{bs}"
    print(f"\n== {name}", flush=True)
    cleanup()
    try:
        model = load(model_id, "lora_bf16").eval()
        vocab = model.config.vocab_size
        with torch.no_grad():
            for _ in range(3):
                model(input_ids=torch.randint(0, vocab, (bs, seq_len), device="cuda"))
            torch.cuda.synchronize(); t0 = time.time()
            for _ in range(N_STEPS):
                model(input_ids=torch.randint(0, vocab, (bs, seq_len), device="cuda"))
            torch.cuda.synchronize(); dt = (time.time() - t0) / N_STEPS
        res = dict(ok=1, tokens_per_s=round(bs * seq_len / dt), seqs_per_s=round(bs / dt, 1), step_ms=round(dt * 1000, 1),
                   peak_alloc_GiB=round(torch.cuda.max_memory_allocated() / 2**30, 2),
                   mfu_pct=round(100 * bs * seq_len / dt * 2 * n_params(model) / (peak * 1e12), 1))
        del model
    except Exception as e:
        res = dict(ok=0, error=f"{type(e).__name__}: {str(e)[:200]}")
    print("  ", res, flush=True); cleanup()
    return name, res


def bench_minilm(bs=32, seq_len=32):
    """Contrastive pairs/s for all-MiniLM-L6-v2 (the embedding recipe used in the experiments)."""
    import torch.nn.functional as F
    from transformers import AutoModel
    name = f"all-MiniLM-L6-v2|contrastive_full_fp32|L{seq_len}|bs{bs}"
    print(f"\n== {name}", flush=True); cleanup()
    model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2").cuda().train()
    opt = torch.optim.AdamW(model.parameters(), lr=3e-5)
    def emb(ids):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            return F.normalize(model(input_ids=ids).last_hidden_state.mean(1), dim=-1).float()
    def step():
        a = emb(torch.randint(0, 30000, (bs, seq_len), device="cuda")); p = emb(torch.randint(0, 30000, (bs, seq_len), device="cuda"))
        loss = F.cross_entropy(a @ p.T * 20, torch.arange(bs, device="cuda")); loss.backward(); opt.step(); opt.zero_grad()
    for _ in range(5): step()
    torch.cuda.synchronize(); t0 = time.time()
    for _ in range(50): step()
    torch.cuda.synchronize(); dt = (time.time() - t0) / 50
    res = dict(ok=1, pairs_per_s=round(bs / dt), step_ms=round(dt * 1000, 1), peak_alloc_GiB=round(torch.cuda.max_memory_allocated() / 2**30, 2))
    print("  ", res, flush=True); del model, opt; cleanup()
    return name, res


configs = [
    # (model, method, seq_len, bs, grad_ckpt, unsloth)
    ("Qwen/Qwen2.5-0.5B", "full_fp32master", 64, 16, False, False),   # the merchant experiment setting
    ("Qwen/Qwen2.5-0.5B", "full_fp32master", 512, 8, False, False),
    ("Qwen/Qwen2.5-0.5B", "lora_bf16", 64, 16, False, False),
    ("Qwen/Qwen2.5-0.5B", "lora_bf16", 512, 8, False, False),
    ("Qwen/Qwen2.5-3B", "lora_bf16", 64, 16, True, False),            # the universe ladder setting
    ("Qwen/Qwen2.5-3B", "lora_bf16", 64, 16, False, False),
    ("Qwen/Qwen2.5-3B", "lora_bf16", 512, 8, True, False),
    ("Qwen/Qwen2.5-3B", "full_fp32master", 64, 8, True, False),       # expected OOM: 16 B/param
    ("Qwen/Qwen2.5-7B", "lora_bf16", 64, 8, True, False),
    ("Qwen/Qwen2.5-7B", "lora_bf16", 512, 4, True, False),
    ("Qwen/Qwen2.5-7B", "qlora_4bit", 512, 4, True, False),
    ("Qwen/Qwen2.5-0.5B", "lora_bf16", 512, 8, False, True),          # unsloth kernels
    ("Qwen/Qwen2.5-3B", "lora_bf16", 512, 8, True, True),
    ("Qwen/Qwen2.5-7B", "qlora_4bit", 512, 4, True, True),
]
if QUICK:
    configs = [c for c in configs if "0.5B" in c[0]]

results = {"gpu": GPU, "peak_bf16_tflops_assumed": peak, "torch": torch.__version__}
with Run("bench_throughput", model="multi", config=dict(n_steps=N_STEPS, quick=QUICK, gpu=GPU, peak_tflops=peak, lora_r=64)) as run:
    for m, method, L, bs, gc_, us in configs:
        name, res = bench_train(m, method, L, bs, grad_ckpt=gc_, unsloth=us)
        results[name] = res; run.log(res, condition=name)
        OUT.write_text(json.dumps(results, indent=2))
    for m, L, bs in (("Qwen/Qwen2.5-0.5B", 128, 16), ("Qwen/Qwen2.5-3B", 128, 16)) + (() if QUICK else (("Qwen/Qwen2.5-7B", 128, 8),)):
        name, res = bench_forward(m, L, bs); results[name] = res; run.log(res, condition=name)
    name, res = bench_minilm(); results[name] = res; run.log(res, condition=name)
    OUT.write_text(json.dumps(results, indent=2)); run.artifact(OUT)

print("\n=== SUMMARY ===")
print(f"{'config':52s}{'tok/s':>9s}{'step ms':>9s}{'peak GiB':>10s}{'MFU %':>7s}")
for k, r in results.items():
    if isinstance(r, dict):
        print(f"{k:52s}{str(r.get('tokens_per_s', r.get('pairs_per_s', ''))):>9s}{str(r.get('step_ms', '')):>9s}"
              f"{str(r.get('peak_alloc_GiB', '')):>10s}{str(r.get('mfu_pct', '')):>7s}" + (f"   {r['error']}" if not r.get("ok") else ""))
