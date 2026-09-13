"""Training / inference throughput and memory benchmark on this GPU.

Each config runs in its own subprocess so a CUDA OOM (which poisons the CUDA
context) is recorded as a data point instead of killing the sweep. For each
(model, method, seq_len, batch) we warm up, time N optimizer steps on synthetic
token batches (isolates the GPU from tokenization), and record tokens/s, step
time, peak VRAM and model-FLOPs utilization (MFU) vs the GPU's dense bf16 peak.
Also: forward-only (likelihood scoring) throughput and the MiniLM contrastive loop.

usage: uv run python experiments/bench_throughput.py [quick]
       uv run python experiments/bench_throughput.py --one <idx> <out.json>   (internal)
Results -> results/bench_throughput.json and the evals tracker (experiment "bench_throughput").
"""
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent.parent
OUT = ROOT / "results" / "bench_throughput.json"
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
PEAK_TFLOPS = {"RTX 3090": 71.0, "RTX 4090": 165.0, "A100": 312.0, "H100": 989.0, "RTX 5090": 210.0}  # dense bf16, fp32 accumulate

# kind, model, method, seq_len, bs, grad_ckpt, unsloth
CONFIGS = [
    ("train", "Qwen/Qwen2.5-0.5B", "full_fp32master", 64, 16, False, False),   # merchant experiment setting
    ("train", "Qwen/Qwen2.5-0.5B", "full_fp32master", 512, 4, False, False),
    ("train", "Qwen/Qwen2.5-0.5B", "lora_bf16", 64, 16, False, False),
    ("train", "Qwen/Qwen2.5-0.5B", "lora_bf16", 512, 8, False, False),
    ("train", "Qwen/Qwen2.5-3B", "lora_bf16", 64, 16, True, False),            # universe ladder setting
    ("train", "Qwen/Qwen2.5-3B", "lora_bf16", 64, 16, False, False),
    ("train", "Qwen/Qwen2.5-3B", "lora_bf16", 512, 8, True, False),
    ("train", "Qwen/Qwen2.5-3B", "full_fp32master", 64, 8, True, False),       # expected OOM: ~16 B/param
    ("train", "Qwen/Qwen2.5-7B", "lora_bf16", 64, 8, True, False),
    ("train", "Qwen/Qwen2.5-7B", "lora_bf16", 512, 4, True, False),
    ("train", "Qwen/Qwen2.5-7B", "qlora_4bit", 512, 4, True, False),
    ("train", "Qwen/Qwen2.5-0.5B", "lora_bf16", 512, 8, False, True),          # unsloth kernels
    ("train", "Qwen/Qwen2.5-3B", "lora_bf16", 512, 8, True, True),
    ("train", "Qwen/Qwen2.5-7B", "qlora_4bit", 512, 4, True, True),
    ("forward", "Qwen/Qwen2.5-0.5B", "bf16", 128, 16, False, False),
    ("forward", "Qwen/Qwen2.5-3B", "bf16", 128, 16, False, False),
    ("forward", "Qwen/Qwen2.5-7B", "bf16", 128, 8, False, False),
    ("minilm", "sentence-transformers/all-MiniLM-L6-v2", "contrastive_fp32master", 32, 32, False, False),
]


def cfg_name(c):
    kind, m, method, L, bs, gc_, us = c
    return f"{m.split('/')[-1]}|{'unsloth_' if us else ''}{method}{'+gc' if gc_ else ''}|L{L}|bs{bs}" + ("|forward" if kind == "forward" else "")


# ============================================================================ child
def run_one(idx, out_path, n_steps):
    import torch
    kind, model_id, method, L, bs, grad_ckpt, unsloth = CONFIGS[idx]
    gpu = torch.cuda.get_device_name(0)
    peak = next((v for k, v in PEAK_TFLOPS.items() if k in gpu), 71.0)
    res = {}
    t_load = time.time()
    try:
        if kind == "minilm":
            import torch.nn.functional as F
            from transformers import AutoModel
            model = AutoModel.from_pretrained(model_id).cuda().train()
            opt = torch.optim.AdamW(model.parameters(), lr=3e-5)
            def emb(ids):
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    return F.normalize(model(input_ids=ids).last_hidden_state.mean(1), dim=-1).float()
            def step():
                a = emb(torch.randint(0, 30000, (bs, L), device="cuda")); p = emb(torch.randint(0, 30000, (bs, L), device="cuda"))
                loss = F.cross_entropy(a @ p.T * 20, torch.arange(bs, device="cuda")); loss.backward(); opt.step(); opt.zero_grad()
            for _ in range(5): step()
            torch.cuda.synchronize(); t0 = time.time()
            for _ in range(50): step()
            torch.cuda.synchronize(); dt = (time.time() - t0) / 50
            res = dict(ok=1, pairs_per_s=round(bs / dt), tokens_per_s=round(2 * bs * L / dt), step_ms=round(dt * 1000, 1),
                       peak_alloc_GiB=round(torch.cuda.max_memory_allocated() / 2**30, 2))
        else:
            from transformers import AutoModelForCausalLM
            if unsloth:
                from unsloth import FastLanguageModel
                model, _ = FastLanguageModel.from_pretrained(model_id, max_seq_length=L, dtype=torch.bfloat16,
                                                             load_in_4bit=(method == "qlora_4bit"))
                base_params = sum(p.numel() for n, p in model.named_parameters())
                if kind == "train":
                    model = FastLanguageModel.get_peft_model(model, r=64, lora_alpha=128, lora_dropout=0.0, target_modules=LORA_TARGETS,
                                                             use_gradient_checkpointing="unsloth" if grad_ckpt else False)
            else:
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
                base_params = sum(p.numel() for p in model.parameters())
                if kind == "train" and method in ("lora_bf16", "qlora_4bit"):
                    from peft import LoraConfig, get_peft_model
                    if method == "qlora_4bit":
                        from peft import prepare_model_for_kbit_training
                        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=grad_ckpt)
                    elif grad_ckpt:
                        model.gradient_checkpointing_enable(); model.enable_input_require_grads()
                    model = get_peft_model(model, LoraConfig(r=64, lora_alpha=128, lora_dropout=0.0, target_modules=LORA_TARGETS))
                elif kind == "train" and grad_ckpt:
                    model.gradient_checkpointing_enable()
            load_s = time.time() - t_load
            vocab = model.config.vocab_size
            if kind == "forward":
                model.eval()
                with torch.no_grad():
                    for _ in range(3): model(input_ids=torch.randint(0, vocab, (bs, L), device="cuda"))
                    torch.cuda.synchronize(); t0 = time.time()
                    for _ in range(n_steps): model(input_ids=torch.randint(0, vocab, (bs, L), device="cuda"))
                    torch.cuda.synchronize(); dt = (time.time() - t0) / n_steps
                res = dict(ok=1, base_params_B=round(base_params / 1e9, 3), tokens_per_s=round(bs * L / dt), seqs_per_s=round(bs / dt, 1),
                           step_ms=round(dt * 1000, 1), peak_alloc_GiB=round(torch.cuda.max_memory_allocated() / 2**30, 2),
                           mfu_pct=round(100 * bs * L / dt * 2 * base_params / (peak * 1e12), 1), load_s=round(load_s, 1))
            else:
                params = [p for p in model.parameters() if p.requires_grad]
                opt = torch.optim.AdamW(params, lr=1e-5)
                model.train()
                autocast = method == "full_fp32master"
                def step():
                    ids = torch.randint(0, vocab, (bs, L), device="cuda")
                    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=autocast):
                        loss = model(input_ids=ids, labels=ids).loss
                    loss.backward(); opt.step(); opt.zero_grad(set_to_none=True)
                for _ in range(3): step()
                torch.cuda.synchronize(); t0 = time.time()
                for _ in range(n_steps): step()
                torch.cuda.synchronize(); dt = (time.time() - t0) / n_steps
                toks = bs * L / dt
                # FLOPs/token: full FT 6N; LoRA ~4N (frozen weight grads skipped); grad ckpt adds ~2N recompute
                fpt = (6 if method == "full_fp32master" else 4) * base_params + (2 * base_params if grad_ckpt else 0)
                res = dict(ok=1, base_params_B=round(base_params / 1e9, 3), trainable_M=round(sum(p.numel() for p in params) / 1e6, 1),
                           step_ms=round(dt * 1000, 1), tokens_per_s=round(toks), samples_per_s=round(bs / dt, 2),
                           peak_alloc_GiB=round(torch.cuda.max_memory_allocated() / 2**30, 2),
                           peak_reserved_GiB=round(torch.cuda.max_memory_reserved() / 2**30, 2),
                           mfu_pct=round(100 * toks * fpt / (peak * 1e12), 1), load_s=round(load_s, 1))
    except Exception as e:
        msg = str(e).splitlines()[0][:160] if str(e) else ""
        res = dict(ok=0, error=f"{type(e).__name__}: {msg}", peak_alloc_GiB=round(torch.cuda.max_memory_allocated() / 2**30, 2))
    Path(out_path).write_text(json.dumps(res))


# ============================================================================ parent
def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--one":
        run_one(int(sys.argv[2]), sys.argv[3], int(sys.argv[4]))
        return
    quick = len(sys.argv) > 1 and sys.argv[1] == "quick"
    n_steps = 6 if quick else 12
    import torch
    sys.path.insert(0, str(ROOT))
    from evals.tracker import Run
    gpu = torch.cuda.get_device_name(0)
    peak = next((v for k, v in PEAK_TFLOPS.items() if k in gpu), 71.0)
    idxs = [i for i, c in enumerate(CONFIGS) if not quick or "0.5B" in c[1] or c[0] == "minilm"]
    results = {"gpu": gpu, "peak_bf16_tflops_assumed": peak, "torch": torch.__version__, "n_steps": n_steps}
    tmp = ROOT / "results" / "_bench_one.json"
    with Run("bench_throughput", model="multi", config=dict(n_steps=n_steps, quick=quick, gpu=gpu, peak_tflops=peak, lora_r=64,
                                                             isolation="subprocess_per_config")) as run:
        for i in idxs:
            name = cfg_name(CONFIGS[i]); print(f"\n== [{i}] {name}", flush=True)
            tmp.unlink(missing_ok=True); t0 = time.time()
            p = subprocess.run([sys.executable, __file__, "--one", str(i), str(tmp), str(n_steps)], capture_output=True, text=True, timeout=1800)
            if tmp.exists():
                res = json.loads(tmp.read_text())
            else:
                tail = (p.stderr or p.stdout).strip().splitlines()[-1:] or ["no output"]
                res = dict(ok=0, error=f"subprocess exit {p.returncode}: {tail[0][:160]}")
            res["wall_s"] = round(time.time() - t0, 1)
            print("  ", res, flush=True)
            results[name] = res; run.log(res, condition=name)
            OUT.write_text(json.dumps(results, indent=2))
        run.artifact(OUT)
    tmp.unlink(missing_ok=True)
    print("\n=== SUMMARY ===")
    print(f"{'config':58s}{'tok/s':>8s}{'step ms':>9s}{'peak GiB':>10s}{'MFU %':>7s}")
    for k, r in results.items():
        if isinstance(r, dict):
            print(f"{k:58s}{str(r.get('tokens_per_s', '')):>8s}{str(r.get('step_ms', '')):>9s}{str(r.get('peak_alloc_GiB', '')):>10s}"
                  f"{str(r.get('mfu_pct', '')):>7s}" + (f"   {r['error']}" if not r.get("ok") else ""))


if __name__ == "__main__":
    main()
