"""Diagnostic (PLAN step 38, INFRA-2): what does unsloth's FastLanguageModel.from_pretrained load for a bf16 base and for a saved
adapter directory? Prints the layer classes and weight dtypes, the recorded base name, quantisation, memory, and the LoRA
weights' scaling against the adapter file, for the base and for the two no-DB categoriser adapters (unsloth-trained, peft-trained).

usage: uv run python scripts/check_unsloth_base.py
"""
import json
import os
import sys

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import unsloth  # noqa: F401
import torch
from safetensors.torch import load_file
from unsloth import FastLanguageModel

from ai_experiments.paths import ROOT

AD = ROOT / "models" / "adapters"
targets = sys.argv[1:] or ["Qwen/Qwen2.5-3B-Instruct", str(AD / "categoriser_Qwen2.5-3B-Instruct_none_lora"), str(AD / "categoriser_Qwen2.5-3B-Instruct_none_hf_lora")]


def describe(src, **kw):
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    model, tok = FastLanguageModel.from_pretrained(src, max_seq_length=2048, dtype=torch.bfloat16, **kw)
    inner = model
    while hasattr(inner, "model") and not hasattr(inner, "layers"):
        inner = inner.model
    layer = inner.layers[0]
    up = layer.mlp.up_proj
    base_lin = getattr(up, "base_layer", up)
    w = getattr(base_lin, "weight", None)
    info = dict(src=src.split("/")[-1], name_or_path=getattr(model.config, "_name_or_path", None), quant=str(getattr(model.config, "quantization_config", None))[:120],
                up_proj_class=type(up).__name__, base_layer_class=type(base_lin).__name__, weight_dtype=str(getattr(w, "dtype", None)), weight_shape=tuple(w.shape) if w is not None else None,
                alloc_GiB=round(torch.cuda.memory_allocated() / 2**30, 2), n_params_M=round(sum(p.numel() for p in model.parameters()) / 1e6))
    if hasattr(up, "lora_B"):
        B = up.lora_B["default"].weight; A = up.lora_A["default"].weight
        info.update(lora_scaling=up.scaling["default"], lora_B_norm=round(B.float().norm().item(), 4), lora_A_norm=round(A.float().norm().item(), 4), lora_r=A.shape[0])
        sd = load_file(os.path.join(src, "adapter_model.safetensors"))
        keys = [k for k in sd if "layers.0.mlp.up_proj.lora_B" in k]
        info.update(file_B_norm=round(sd[keys[0]].float().norm().item(), 4) if keys else None, file_key=keys[0] if keys else None,
                    adapter_base=json.load(open(os.path.join(src, "adapter_config.json")))["base_model_name_or_path"])
    # one forward: perplexity of a fixed sentence, as a fingerprint of the loaded weights
    ids = tok("The quick brown fox jumps over the lazy dog because it has somewhere to be.", return_tensors="pt")["input_ids"].cuda()
    with torch.no_grad():
        logits = model(input_ids=ids).logits.float()
    info["ppl"] = round(torch.exp(torch.nn.functional.cross_entropy(logits[0, :-1], ids[0, 1:])).item(), 4)
    print(json.dumps(info, indent=1), flush=True)
    del model; torch.cuda.empty_cache()


for t in targets:
    describe(t)
print("--- base with load_in_4bit=False explicitly")
describe(targets[0], load_in_4bit=False)
