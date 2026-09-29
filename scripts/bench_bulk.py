"""Bulk inference for one user's sync (PLAN step 119, INFRA-3; owner 2026-09-29: one user's 10 to 100 new transactions at once). On one
H100, decider-4B with the recipe's adapter (labelled rows, REPORT 113), per sync of blind_bulk_v1 (one user, up to 30 new transactions):

  sep        each transaction its own prompt in today's layout (rows of the slice in date order), batched BATCH at a time
  split      each transaction its own prompt in the split layout (the sync's shared rows first, then its payee's), batched BATCH at a time
  cached     the split layout with the sync's common token prefix run once and every transaction's tail run from that cache, all tails in
             one batch; its answers must equal `split`'s (the same tokens), which is the correctness check
  multi      one sequence: the common prefix, then each transaction's tail with its own answer slot, "?)" after each slot; one pass reads
             every slot (zero-shot: the adapter never saw unanswered transactions before the one it reads)

transformers 5.17's Qwen3.5 linear-attention layer continues a multi-token forward from the cached conv and recurrent states (older
releases restarted them, which would make `cached` silently wrong); the check against `split` guards it.
Labels and option order are drawn once per sync (production would fix them per user), so every prompt of a sync starts with the same
tokens. Timing: warm (one untimed pass per sync and mode), CUDA-synchronised, median of REPEAT.
Writes results/bench_bulk_<adapter>.json (per sync and mode: ms, tokens, per-item log-probs) and prints the summary.
usage: ADAPTER=<name> uv run --with transformers==5.17.0 --with flash-linear-attention --with "peft>=0.21" --with torch==2.13.0 ... python scripts/bench_bulk.py
"""
import importlib
import json
import os
import random
import statistics
import sys
import time

from ai_experiments.paths import PROCESSED, ROOT

MODEL = os.environ.get("MODEL", "Mapika/decider-4b")
ADAPTER = os.environ.get("ADAPTER", "")
N_SYNCS = int(os.environ.get("N_SYNCS", "24"))
SIZES = [int(x) for x in os.environ.get("SIZES", "10,30").split(",")]
BATCH = int(os.environ.get("BATCH", "8"))
REPEAT = int(os.environ.get("REPEAT", "3"))
QUESTION = "Which of this user's categories does the last transaction belong to?"


def expand_cache(cache, n):
    """The batch-1 prefix cache repeated n times along the batch, through the cache's own beam-search reorder (index 0, n times), which
    covers attention keys / values and the linear-attention conv and recurrent states however the release stores them."""
    import torch
    cache.reorder_cache(torch.zeros(n, dtype=torch.long, device="cuda"))
    return cache


def main():
    import torch
    import torch.nn.functional as F
    from huggingface_hub import snapshot_download
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from ai_experiments import oneslot
    path = snapshot_download(MODEL)
    sys.path.insert(0, snapshot_download("Mapika/decider-2b", allow_patterns=["decider/*"]))
    P = importlib.import_module("decider.prompt")
    tok = AutoTokenizer.from_pretrained(path)
    lm = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16).cuda().eval()
    if ADAPTER:
        from peft import PeftModel
        lm = PeftModel.from_pretrained(lm, str(ROOT / "models" / "adapters" / ADAPTER)).merge_and_unload().eval()
    head = lm.lm_head.weight
    pad = tok.pad_token_id or 0

    plain = json.loads((PROCESSED / "blind_bulk_v1.json").read_text())["items"]
    split = {i["id"]: i for i in json.loads((PROCESSED / "blind_bulk_v1_split.json").read_text())["items"]}
    syncs = {}
    for it in plain:
        syncs.setdefault(it["user"], []).append(it)
    users = sorted(syncs)[:: max(1, len(syncs) // N_SYNCS)][:N_SYNCS]

    def built(it, layout_item):
        st = layout_item["prompt"][:-len("Category:")].rstrip()
        return oneslot.build_layout(P, tok, st, QUESTION, [o.strip() for o in it["options"]], it["answer"], random.Random(f"sync-{it['user']}"),
                                    labels="rand255", layout="labelled_shots")

    def readout(h, slots, labs):
        return [F.log_softmax(F.linear(h[i, s], head[torch.tensor(labs, device="cuda")]).float(), -1).tolist() for i, s in enumerate(slots)]

    @torch.no_grad()
    def run_sep(bs):  # each prompt on its own, BATCH per forward, lengths rounded up to 64 as the scorer does
        out = []
        for k in range(0, len(bs), BATCH):
            ch = bs[k:k + BATCH]; T = -(-max(len(b["ids"]) for b in ch) // 64) * 64
            ids = torch.full((len(ch), T), pad, dtype=torch.long); att = torch.zeros_like(ids)
            for i, b in enumerate(ch):
                ids[i, :len(b["ids"])] = torch.tensor(b["ids"]); att[i, :len(b["ids"])] = 1
            h = lm.model(input_ids=ids.cuda(), attention_mask=att.cuda()).last_hidden_state
            out += readout(h, [b["slot"] for b in ch], ch[0]["labs"])
        return out

    def lcp(bs):
        n = min(len(b["ids"]) for b in bs)
        for j in range(n):
            if any(b["ids"][j] != bs[0]["ids"][j] for b in bs):
                return j
        return n

    @torch.no_grad()
    def run_cached(bs, L):
        pre = torch.tensor([bs[0]["ids"][:L]], device="cuda")
        o = lm.model(input_ids=pre, use_cache=True)
        cache = expand_cache(o.past_key_values, len(bs))
        tails = [b["ids"][L:] for b in bs]; T = -(-max(map(len, tails)) // 64) * 64
        ids = torch.full((len(bs), T), pad, dtype=torch.long); att = torch.zeros((len(bs), L + T), dtype=torch.long); att[:, :L] = 1
        for i, t in enumerate(tails):
            ids[i, :len(t)] = torch.tensor(t); att[i, L:L + len(t)] = 1
        h = lm.model(input_ids=ids.cuda(), attention_mask=att.cuda(), past_key_values=cache, use_cache=True).last_hidden_state
        return readout(h, [b["slot"] - L for b in bs], bs[0]["labs"])

    close = tok.encode("?)\n\n", add_special_tokens=False)

    @torch.no_grad()
    def run_multi(bs, L):
        seq, slots = list(bs[0]["ids"][:L]), []
        for b in bs:
            seq += b["ids"][L:]; slots.append(len(seq) - 1); seq += close
        h = lm.model(input_ids=torch.tensor([seq], device="cuda")).last_hidden_state
        return [F.log_softmax(F.linear(h[0, s], head[torch.tensor(bs[0]["labs"], device="cuda")]).float(), -1).tolist() for s in slots], len(seq)

    def timed(fn):
        fn(); torch.cuda.synchronize()
        ts = []
        for _ in range(REPEAT):
            t0 = time.perf_counter(); r = fn(); torch.cuda.synchronize(); ts.append(time.perf_counter() - t0)
        return r, 1000 * statistics.median(ts)

    recs = []
    for u in users:
        its = sorted(syncs[u], key=lambda x: x["pos"])
        for n in SIZES:
            sub = its[:n]
            b_sep = [built(it, it) for it in sub]; b_spl = [built(it, split[it["id"]]) for it in sub]
            assert all(b["labs"] == b_spl[0]["labs"] for b in b_spl + b_sep)
            L = lcp(b_spl)
            r_sep, t_sep = timed(lambda: run_sep(b_sep))
            r_spl, t_spl = timed(lambda: run_sep(b_spl))
            r_cac, t_cac = timed(lambda: run_cached(b_spl, L))
            (r_mul, n_mul), t_mul = timed(lambda: run_multi(b_spl, L))
            # answers back in the items' option order
            back = lambda lp, b: [lp[b["perm"].index(o)] for o in range(len(b["perm"]))]  # noqa: E731
            rec = dict(user=u, n=len(sub), prefix=L, tok_sep=sum(len(b["ids"]) for b in b_sep), tok_split=sum(len(b["ids"]) for b in b_spl),
                       tok_cached=L + sum(len(b["ids"]) - L for b in b_spl), tok_multi=n_mul,
                       ms=dict(sep=t_sep, split=t_spl, cached=t_cac, multi=t_mul), ids=[it["id"] for it in sub], answer=[it["answer"] for it in sub],
                       lp={m: [back(lp, b) for lp, b in zip(r, bb)] for m, r, bb in (("sep", r_sep, b_sep), ("split", r_spl, b_spl), ("cached", r_cac, b_spl), ("multi", r_mul, b_spl))})
            recs.append(rec)
            d = max(abs(x - y) for a, c in zip(rec["lp"]["split"], rec["lp"]["cached"]) for x, y in zip(a, c))
            print(f"user {u} n={len(sub)} prefix {L} tok sep/split/cached/multi {rec['tok_sep']}/{rec['tok_split']}/{rec['tok_cached']}/{n_mul} "
                  f"ms {t_sep:.0f}/{t_spl:.0f}/{t_cac:.0f}/{t_mul:.0f}; cached vs split max |dlogp| {d:.3f}", flush=True)
    out = ROOT / "results" / f"bench_bulk_{ADAPTER or MODEL.split('/')[-1]}.json"
    out.write_text(json.dumps(dict(model=MODEL, adapter=ADAPTER, batch=BATCH, repeat=REPEAT, gpu=torch.cuda.get_device_name(), recs=recs)))
    import numpy as np
    for n in SIZES:
        rs = [r for r in recs if r["n"] == n]
        if not rs:
            continue
        per = {m: np.mean([r["ms"][m] / r["n"] for r in rs]) for m in ("sep", "split", "cached", "multi")}
        acc = {m: 100 * np.mean([int(np.argmax(lp)) == a for r in rs for lp, a in zip(r["lp"][m], r["answer"])]) for m in ("sep", "split", "cached", "multi")}
        agree = 100 * np.mean([int(np.argmax(x)) == int(np.argmax(y)) for r in rs for x, y in zip(r["lp"]["split"], r["lp"]["cached"])])
        print(f"== n={n} ({len(rs)} syncs): ms per transaction " + ", ".join(f"{m} {v:.1f}" for m, v in per.items())
              + " | top-1 " + ", ".join(f"{m} {v:.1f}" for m, v in acc.items()) + f" | cached = split on {agree:.1f}%", flush=True)


if __name__ == "__main__":
    main()
