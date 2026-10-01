"""PLAN step 127 (b) (INFRA-3): one user's sync served by vLLM, with and without its prefix cache, on the prompts `bench_bulk.py PREP_VLLM=<dir>`
wrote (the merged decider-4B recipe model and, per sync, every transaction's token ids in the split and in today's layout). Standalone
(numpy and vllm only), run in its own environment: `uv run --no-project --with vllm python scripts/bench_vllm.py <dir>`.

The readout is decider's one slot: one generated token restricted to the question's label tokens (`allowed_token_ids`), its log-probs
after that restriction (`logprobs_mode="processed_logprobs"`), which is the softmax over the label logits the repository reads.
Arms: prefix caching off / on; per sync (latency: one sync submitted at a time, 10 and 30 transactions) and all syncs at once
(throughput). Answers are compared with the HF uncached split layout from results/bench_bulk_*.json when present.
Writes results/bench_vllm.json.
"""
import glob
import json
import sys
import time
from pathlib import Path

import numpy as np


def main(d):
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt
    import vllm
    syncs = json.loads(Path(d, "syncs.json").read_text())
    ref = {}
    for f in glob.glob("results/bench_bulk_*.json"):
        for r in json.loads(Path(f).read_text())["recs"]:
            for i, lp in zip(r["ids"], r["lp"]["split"]):
                ref[i] = int(np.argmax(lp))
    out = dict(vllm=vllm.__version__, arms={})
    for cache in (False, True):
        try:
            llm = LLM(model=str(Path(d, "model")), dtype="bfloat16", enable_prefix_caching=cache, max_logprobs=300, logprobs_mode="processed_logprobs",
                      gpu_memory_utilization=0.85, max_model_len=8192, seed=0)
        except Exception as e:  # noqa: BLE001
            print(f"prefix caching {cache}: engine failed: {type(e).__name__}: {e}", flush=True)
            out["arms"][f"cache{int(cache)}"] = dict(error=f"{type(e).__name__}: {e}"[:500]); continue

        def run(prompts, labs_list):
            sps = [SamplingParams(max_tokens=1, temperature=0.0, logprobs=len(l), allowed_token_ids=list(l)) for l in labs_list]
            res = llm.generate([TokensPrompt(prompt_token_ids=p) for p in prompts], sps, use_tqdm=False)
            lps = []
            for r, labs in zip(res, labs_list):
                d0 = r.outputs[0].logprobs[0]
                lps.append([d0[t].logprob if t in d0 else -1e9 for t in labs])
            return lps
        arm = {}
        for layout in ("split", "sep"):
            for n in (10, 30):
                ms, agree, right = [], [], []
                for s in syncs:
                    prompts = s[layout][:n]; labs = [s["labs"]] * len(prompts)
                    run(prompts, labs)  # warm: fills the prefix cache for this sync (the second sync onwards is also warm)
                    llm.reset_prefix_cache() if cache else None
                    t0 = time.perf_counter(); lps = run(prompts, labs); ms.append(1000 * (time.perf_counter() - t0) / len(prompts))
                    for i, lp, perm, a in zip(s["ids"][:n], lps, s["perm"][:n], s["answer"][:n]):
                        pick = perm[int(np.argmax(lp))]; right.append(pick == a)
                        if i in ref and layout == "split":
                            agree.append(pick == ref[i])
                arm[f"{layout}_n{n}"] = dict(ms_per_txn=float(np.mean(ms)), top1=100 * float(np.mean(right)), same_as_hf=100 * float(np.mean(agree)) if agree else None)
                print(f"cache={cache} {layout} n={n}: {np.mean(ms):.1f} ms per transaction, top-1 {100 * np.mean(right):.1f}"
                      + (f", same as HF {100 * np.mean(agree):.1f}%" if agree else ""), flush=True)
            prompts = [p for s in syncs for p in s[layout]]; labs = [s["labs"] for s in syncs for _ in s[layout]]
            run(prompts[:64], labs[:64]); llm.reset_prefix_cache() if cache else None
            t0 = time.perf_counter(); run(prompts, labs); t = time.perf_counter() - t0
            arm[f"{layout}_all"] = dict(ms_per_txn=1000 * t / len(prompts), n=len(prompts))
            print(f"cache={cache} {layout} all {len(prompts)} at once: {1000 * t / len(prompts):.1f} ms per transaction", flush=True)
        out["arms"][f"cache{int(cache)}"] = arm
        del llm
        import gc
        import torch
        gc.collect(); torch.cuda.empty_cache()
    Path("results/bench_vllm.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main(sys.argv[1])
