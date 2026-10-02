"""PLAN step 158 (owner, 2026-10-02: "I definitely want a user's history to perturb the embeddings! Of both their categories and the
payees!"). Personal embeddings: decider's hidden state at the last token of a cue ("In one word, the kind of spending:", final layer;
REPORT 152's best read-out), with the user's history slice in front of the text, for each of the user's categories
('Budget category "<name>"') and for the transaction to file ('Transaction: <date | payee | amount | day>'); against global ones with no
history in front (the category name alone; the name with the payees the user filed under it, REPORT 152's text; the transaction alone).
For each blind item: an embedding-only categoriser (the user's category nearest the transaction, cosine) in every pairing, and how the
history moves the transaction's embedding (cosine to itself without history; its similarity to the gold category with and without).
The history is encoded once per item and its cache forked across the item's texts (strands_decider.infer._fork_layered_cache, Apache-2.0,
which forks Qwen3.5's recurrent states too); the first item is checked against full-sequence encoding.
env: EMB_MODEL (Mapika/decider-4b), ADAPTER (a LoRA under models/adapters, merged), SET (blind_v2), LIMIT (0 = all items).
Writes results/personal_embed_<reader>_<set>.json (per item: the rank of the gold category under each pairing, the movement measures).
usage: EMB_MODEL=Mapika/decider-4b uv run --with strands-decider==0.1.0 --with transformers==5.17.0 --with flash-linear-attention
       --with "peft>=0.21" --with torch==2.13.0 --with torchvision==0.28.0 python scripts/exp_personal_embed.py
"""
import json
import os
import re
from collections import Counter

import numpy as np

from ai_experiments import oneslot
from ai_experiments.paths import PROCESSED, ROOT

EMB_MODEL = os.environ.get("EMB_MODEL", "Mapika/decider-4b")
ADAPTER = os.environ.get("ADAPTER", "")
SET = os.environ.get("SET", "blind_v2")
LIMIT = int(os.environ.get("LIMIT", "0"))
CUE = "\nIn one word, the kind of spending:"
READER = ("recipe" if ADAPTER else EMB_MODEL.split("/")[-1])


def split_prompt(prompt):
    """(history text, query transaction line, category names, {category: filed payee names}) from a blind item's plain prompt."""
    body = prompt.rsplit("\nCategory:", 1)[0]
    hist, _, q = body.rpartition("\n\nTransaction: ")
    names, rows, _ = oneslot.parse(body)
    filed = {}
    for fields, label, _ in rows:
        if label:
            p = oneslot.payee_name(fields)
            if p and p not in filed.setdefault(label, []):
                filed[label].append(p)
    return hist, "Transaction: " + q, names, filed


def main():
    import torch
    from strands_decider.infer import _fork_layered_cache
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(EMB_MODEL)
    lm = AutoModelForCausalLM.from_pretrained(EMB_MODEL, dtype=torch.bfloat16).cuda().eval()
    if ADAPTER:
        from peft import PeftModel
        lm = PeftModel.from_pretrained(lm, str(ROOT / "models" / "adapters" / ADAPTER)).merge_and_unload().eval()
    body = lm.model
    cue = tok(CUE, add_special_tokens=False)["input_ids"]
    pad = tok.pad_token_id or 0

    def enc(t):
        return tok(t, add_special_tokens=False)["input_ids"] + cue

    @torch.no_grad()
    def plain(texts, bs=32):
        out = []
        for k in range(0, len(texts), bs):
            seqs = [enc(t) for t in texts[k:k + bs]]
            T = -(-max(len(s) for s in seqs) // 64) * 64
            ids = torch.full((len(seqs), T), pad, dtype=torch.long); att = torch.zeros_like(ids)
            for i, s in enumerate(seqs):
                ids[i, :len(s)] = torch.tensor(s); att[i, :len(s)] = 1
            h = body(input_ids=ids.cuda(), attention_mask=att.cuda()).last_hidden_state.float()
            out += [h[i, len(s) - 1].cpu().numpy() for i, s in enumerate(seqs)]
        return _norm(np.stack(out))

    @torch.no_grad()
    def with_prefix(prefix, texts):
        """The history encoded once, its cache forked across the texts; each text's cued last-token state."""
        pids = tok(prefix, add_special_tokens=False)["input_ids"]
        P = torch.tensor([pids], device="cuda")
        po = body(input_ids=P, attention_mask=torch.ones_like(P), use_cache=True)
        seqs = [tok("\n\n" + t, add_special_tokens=False)["input_ids"] + cue for t in texts]
        T = max(len(s) for s in seqs)
        ids = torch.full((len(seqs), T), pad, dtype=torch.long); att = torch.zeros_like(ids)
        for i, s in enumerate(seqs):
            ids[i, :len(s)] = torch.tensor(s); att[i, :len(s)] = 1
        cache = _fork_layered_cache(po.past_key_values, len(seqs))
        full = torch.cat([torch.ones(len(seqs), P.shape[1], dtype=att.dtype), att], 1).cuda()
        h = body(input_ids=ids.cuda(), attention_mask=full, past_key_values=cache, use_cache=True).last_hidden_state.float()
        return _norm(np.stack([h[i, len(s) - 1].cpu().numpy() for i, s in enumerate(seqs)])), seqs, pids

    items = json.loads((PROCESSED / f"{SET}.json").read_text())["items"]
    items = items[:LIMIT] if LIMIT else items
    recs, checked = [], False
    for it in items:
        hist, q, names, filed = split_prompt(it["prompt"])
        opts = [o.strip() for o in it["options"]]
        cat_texts = [f'Budget category "{c}"' for c in opts]
        cat_texts_filed = [f'Budget category "{c}"' + (f". Payees filed under it: {', '.join(filed.get(c, [])[:8])}" if filed.get(c) else "") for c in opts]
        Pp, seqs, pids = with_prefix(hist, cat_texts + [q])
        if not checked:  # the forked cache against one full-sequence encoding
            ref = plain([hist + "\n\n" + q])[0]
            agree = float(ref @ Pp[-1]); print(f"   fork check: cosine {agree:.4f} between forked and full encoding", flush=True)
            assert agree > 0.99, agree
            checked = True
        G = plain(cat_texts + cat_texts_filed + [q])
        n = len(opts)
        cp, tp = Pp[:n], Pp[n]
        cg, cgf, tg = G[:n], G[n:2 * n], G[2 * n]
        rank = lambda t, C: int(np.where(np.argsort(-(C @ t)) == it["answer"])[0][0]) + 1  # noqa: E731
        recs.append(dict(id=it["id"], user=it["user"], why=it["why"], first=not it.get("payee_hist"), n=n,
                         ranks={"global txn / name": rank(tg, cg), "global txn / name + filed": rank(tg, cgf), "personal txn / personal cat": rank(tp, cp),
                                "personal txn / name": rank(tp, cg), "personal txn / name + filed": rank(tp, cgf), "global txn / personal cat": rank(tg, cp)},
                         move=float(tp @ tg), gold_sim_global=float(tg @ cg[it["answer"]]), gold_sim_personal=float(tp @ cp[it["answer"]]),
                         gold_margin_global=float(tg @ cg[it["answer"]] - np.max(np.delete(cg @ tg, it["answer"]))) if n > 1 else 0.0,
                         gold_margin_personal=float(tp @ cp[it["answer"]] - np.max(np.delete(cp @ tp, it["answer"]))) if n > 1 else 0.0))
        if len(recs) % 200 == 0:
            print(f"   {len(recs)} items", flush=True)
    return recs


def _norm(X):
    return X / np.clip(np.linalg.norm(X, axis=1, keepdims=True), 1e-9, None)


if __name__ == "__main__":
    from ai_experiments.licences import open_licence
    open_licence(EMB_MODEL)
    recs = main()
    for key in recs[0]["ranks"]:
        r = np.array([x["ranks"][key] for x in recs])
        f = np.array([x["first"] for x in recs])
        print(f"{READER} {key}: right first {100 * np.mean(r == 1):.1f}, top 3 {100 * np.mean(r <= 3):.1f}; first-time payees {100 * np.mean(r[f] == 1):.1f} / "
              f"{100 * np.mean(r[f] <= 3):.1f}; payees seen before {100 * np.mean(r[~f] == 1):.1f}", flush=True)
    print(f"movement: cosine(personal, global txn) mean {np.mean([x['move'] for x in recs]):.3f}; gold-category margin global "
          f"{np.mean([x['gold_margin_global'] for x in recs]):+.3f} -> personal {np.mean([x['gold_margin_personal'] for x in recs]):+.3f}", flush=True)
    (ROOT / "results" / f"personal_embed_{READER}_{SET}.json").write_text(json.dumps(dict(reader=READER, model=EMB_MODEL, adapter=ADAPTER, set=SET, items=recs)))
