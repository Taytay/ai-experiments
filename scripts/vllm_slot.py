"""PLAN step 208, E3 (reports/workflow_review_2026-10-05.md item 1; §120 Serving the sync with vLLM measured 24 ms per item against 53 for
the HF loop, top-1 equal): decider's one-slot readout served by vLLM. Standalone (numpy and vllm only), run in its own environment by
scripts/exp_decision_models.py READER=vllm:
  uv run --no-project --with vllm python scripts/vllm_slot.py <model dir> <prompts.json> <out.json>
prompts.json: [{"ids": [token ids up to and including the answer slot], "labs": [label token ids]}, ...]. One generated token
restricted to the item's label tokens (allowed_token_ids), its log-probs after that restriction (logprobs_mode="processed_logprobs"):
the softmax over the label logits that the HF reader computes. Prefix caching on, items submitted in the given order (household order,
so shared prompt prefixes are reused). out.json: one list of log-probs per item, in label order.
env: VLLM_MEM (0.85), VLLM_MAXLEN (8192).
"""
import json
import os
import sys
import time


def main(model, src, dst):
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt
    items = json.load(open(src))
    t0 = time.time()
    llm = LLM(model=model, dtype="bfloat16", enable_prefix_caching=True, max_logprobs=300, logprobs_mode="processed_logprobs",
              gpu_memory_utilization=float(os.environ.get("VLLM_MEM", "0.85")), max_model_len=int(os.environ.get("VLLM_MAXLEN", "8192")), seed=0)
    t1 = time.time()
    sps = [SamplingParams(max_tokens=1, temperature=0.0, logprobs=len(it["labs"]), allowed_token_ids=list(it["labs"])) for it in items]
    res = llm.generate([TokensPrompt(prompt_token_ids=it["ids"]) for it in items], sps, use_tqdm=False)
    out = []
    for r, it in zip(res, items):
        d0 = r.outputs[0].logprobs[0]
        out.append([d0[t].logprob if t in d0 else -1e9 for t in it["labs"]])
    json.dump(out, open(dst, "w"))
    t2 = time.time()
    print(f"[vllm_slot] {len(items)} items: engine {t1 - t0:.0f}s, scoring {t2 - t1:.0f}s ({1000 * (t2 - t1) / max(1, len(items)):.1f} ms per item)", flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:4])
