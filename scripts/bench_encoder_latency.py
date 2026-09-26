"""Latency micro-benchmark for the row 53 encoder (ModernBERT-large + Laya's head): the same input repeated, then varying lengths, with
and without the head, to find where time goes. usage: uv run python scripts/bench_encoder_latency.py
"""
import os
import sys
import time

import torch

sys.argv = [sys.argv[0]]
os.environ.setdefault("INIT", "mbert")
sys.path.insert(0, os.path.dirname(__file__))
import exp_encoder_mask as E  # noqa: E402

tok, model = E.load_model(); model.eval()
print("encoder dtype", next(model.encoder.parameters()).dtype, "device", next(model.parameters()).device, "attn", model.encoder.config._attn_implementation,
      "compile", getattr(model.encoder.config, "reference_compile", None), flush=True)


def t(fn, n=20):
    for _ in range(3):
        fn()
    torch.cuda.synchronize(); t0 = time.time()
    for _ in range(n):
        fn()
    torch.cuda.synchronize(); return 1000 * (time.time() - t0) / n


for L in (256, 700, 1400):
    ids = torch.randint(1000, 20000, (1, L)).cuda(); att = torch.ones_like(ids); pos = torch.arange(8)[None].cuda() * 10 + 5; pm = torch.ones_like(pos, dtype=torch.bool)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        full = t(lambda: model(ids, att, pos, pm)); enc = t(lambda: model.encoder(input_ids=ids, attention_mask=att))
    model_bf = model.to(torch.bfloat16)
    with torch.no_grad():
        pure = t(lambda: model_bf(ids, att, pos, pm))
    model.float()
    print(f"len {L}: full model autocast {full:.1f} ms, encoder only {enc:.1f} ms, full model pure bf16 {pure:.1f} ms", flush=True)
ids = torch.randint(1000, 20000, (32, 700)).cuda(); att = torch.ones_like(ids); pos = (torch.arange(8)[None] * 10 + 5).expand(32, -1).cuda(); pm = torch.ones_like(pos, dtype=torch.bool)
with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
    print(f"batch 32 x 700: {t(lambda: model(ids, att, pos, pm), 5) / 32:.2f} ms per item", flush=True)
