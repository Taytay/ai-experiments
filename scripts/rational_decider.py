"""Row 234 (owner, 2026-10-06: "I'm surprised it didn't get 100% on those. What does decider do with that budget?"): decider-4B read on the
rational households (scripts/rational_budgets.py) and their name variations, exactly as it reads the owner's budget (real_budget_eval:
SIM=2 GROUPNAMES=1, the split layout, the v5 adapter of row 231), locally. Each household's payee names are embedded with the same
model (real_budget_eval.embed_stream's rendering; the owner's embeddings came from it too) into a per-household folder, its items built
by real_budget_eval.build_items, and every first purchase at a merchant plus every KNOWN_EVERY-th repeat purchase scored. Synthetic
data only: items and scores under data/interim/rational_decider (gitignored).
env: RATIONAL_FILE (rational_variants_v1.json: the six frozen versions; else built here), VARIANTS (": obvious: :synonyms obvious:synonyms", payees:categories as r234_rational.sh), ADAPTER_NAME (decider v5 seed 0), KNOWN_EVERY (4)
usage: uv run --with transformers==5.17.0 --with flash-linear-attention --with "peft>=0.21" --with torch==2.13.0 --with torchvision==0.28.0 \
       python scripts/rational_decider.py
"""
import io
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from ai_experiments.paths import ROOT  # noqa: E402

ADAPTER_NAME = os.environ.get("ADAPTER_NAME", "decider_decider-4b_none_h100bf16st800_emp20_f0_ren50_dbep50_mislead_v1_alt10s_lk10_ov10_oth50_aux100_labrand255_laylabelled_shots_ev10soft_rs25v5g_lora")
os.environ.update(BUDGET="rational", SIM="2", GROUPNAMES="1", READER=f"adapter:{ADAPTER_NAME}@r231-dv5-s0")
os.environ.setdefault("LAYOUT", "today")  # one prompt each, BATCH (16) of similar length a pass: about one transaction a day here, so the
os.environ.setdefault("BATCH", "16")  # split layout's per-day shared prefix saves nothing and runs one item a pass (2 items/s on an H100)
import real_budget_eval as RB  # noqa: E402

OUT = ROOT / "data" / "interim" / "rational_decider"
KNOWN_EVERY = int(os.environ.get("KNOWN_EVERY", "4"))


def _with_account(b):
    return dict(b, accounts=[dict(id="a", name="Checking", deleted=False)], transactions=[dict(t, account_id="a") for t in b["transactions"]])


def embed_all(budgets_by_variant):
    """every household's payee names through decider's embedding (REPORT 152), saved as <OUT>/<budget id>/payee_emb.npz"""
    import torch
    from huggingface_hub import snapshot_download
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    path = snapshot_download("Mapika/decider-4b")
    tok = AutoTokenizer.from_pretrained(path)
    lm = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16).cuda().eval()
    lm = PeftModel.from_pretrained(lm, str(ROOT / "models" / "adapters" / ADAPTER_NAME)).merge_and_unload().eval()
    cue = tok(RB.CUE, add_special_tokens=False)["input_ids"]
    for bs in budgets_by_variant.values():
        for b in bs:
            d = OUT / b["id"]
            if (d / "payee_emb.npz").exists():
                continue
            names = [p["name"] for p in b["payees"]]
            vecs = []
            with torch.no_grad():
                for k in range(0, len(names), 64):
                    seqs = [tok(RB.EMB_TEXT.format(n), add_special_tokens=False)["input_ids"] + cue for n in names[k:k + 64]]
                    T = max(map(len, seqs))
                    ids = torch.full((len(seqs), T), tok.pad_token_id or 0, dtype=torch.long); att = torch.zeros_like(ids)
                    for i, q in enumerate(seqs):
                        ids[i, :len(q)] = torch.tensor(q); att[i, :len(q)] = 1
                    h = lm.model(input_ids=ids.cuda(), attention_mask=att.cuda()).last_hidden_state.float()
                    vecs += [h[i, len(q) - 1].cpu().numpy() for i, q in enumerate(seqs)]
            d.mkdir(parents=True, exist_ok=True)
            np.savez(d / "payee_emb.npz", names=np.array(names), vecs=np.stack(vecs).astype(np.float16))
    del lm
    torch.cuda.empty_cache()


def main():
    import importlib
    variants = os.environ.get("VARIANTS", ": obvious: :synonyms obvious:synonyms").split()
    by = {}
    if os.environ.get("RATIONAL_FILE"):  # the frozen versions (data/processed/rational_variants_v1.json, DVC): the same budgets fcr read
        from ai_experiments.paths import PROCESSED
        doc = json.loads((PROCESSED / os.environ["RATIONAL_FILE"]).read_text())["budgets"]
        by = {v: [_with_account(b) for b in doc[v]] for v in variants}
    for v in ([] if by else variants):
        p, c = v.split(":")
        os.environ["RATIONAL_PAYEES"], os.environ["RATIONAL_CATS"] = p, c
        import rational_budgets as RG
        importlib.reload(RG)
        by[v] = [_with_account(b) for b in RG.budgets("bank")]
    t0 = time.time()
    embed_all(by)
    print(f"payees embedded ({time.time() - t0:.0f}s)", flush=True)
    rows = []
    for v, bs in by.items():
        todo, seg = [], {}
        for b in bs:
            RB.OUT = OUT / b["id"]
            reason = {t["id"]: t["reason"][0] for t in b["transactions"]}
            k = 0
            for it in RB.build_items(b):
                r = reason.get(it["id"])
                if it["answer"] < 0 or r is None:
                    continue
                if r == "known":
                    k += 1
                    if k % KNOWN_EVERY:
                        continue
                todo.append(it); seg[it["id"]] = r
        if os.environ.get("ITEM_LIMIT"):  # a smoke test: every k-th item
            todo = todo[::max(1, len(todo) // int(os.environ["ITEM_LIMIT"]))]
        f = OUT / f"scores_{v.replace(':', '_') or 'base'}.jsonl"
        t1 = time.time()
        with open(f, "w") as fo:
            RB.run_scoring(todo, fo)
        sc = {r["id"]: r for r in map(json.loads, open(f))}
        ans = {it["id"]: it["answer"] for it in todo}
        ok = {i: int(np.argmax(sc[i]["lp"])) == ans[i] for i in sc}
        res = {s: 100 * np.mean([ok[i] for i in ok if seg[i] == s]) for s in ("new", "known")}
        rows.append((v, len(todo), res, time.time() - t1))
        print(f"{v}: {len(todo)} items, new {res['new']:.1f}, known {res['known']:.1f} ({time.time() - t1:.0f}s)", flush=True)
    print("\n| payees | categories | new | known (every 4th) | items |\n|---|---|---|---|---|")
    for v, n, res, _ in rows:
        p, c = v.split(":")
        print(f"| {p or 'held-out real'} | {c or 'as named'} | {res['new']:.1f} | {res['known']:.1f} | {n} |")


if __name__ == "__main__":
    main()
