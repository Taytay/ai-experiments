"""PLAN step 159 (owner, 2026-10-02: "Can you use embeddings to map canonical categories to a personal set of categories or to map two
users' categories together?"). On blind_v2's users (kindcat_v2: 2,867 (user, category) pairs, gold = what the category holds: one of 41
canonical kinds, "several kinds of spending", or "a person, trip or purpose"; ai_experiments.canon), with REPORT 152's read-out (the
hidden state at the last token of "In one word, the kind of spending:", final layer):
  canonical   each user category to the nearest of the 43 canonical options, each embedded as 'Budget category "<option>"'; accuracy
              (right first, in the top 3) against the gold, by level (rows of the category in the slice) and by what the category is
  user-user   for pairs of users, each category of A (one canonical kind) to B's nearest category; right when B's category has the same
              kind, counted where B has one; 600 pairs drawn with a fixed seed
Category embeddings three ways: the name alone; the name with the payees the user filed under it (REPORT 152); personal (the user's
history slice in front of 'Budget category "<name>"', REPORT 153). No-model baselines (BASELINE=1, CPU): character n-gram TF-IDF of the
name (and for canonical, of the option label).
env: EMB_MODEL (Mapika/decider-4b | Qwen/Qwen3-Embedding-4B), ADAPTER, BASELINE. Writes results/category_mapping_<reader>.json.
usage: EMB_MODEL=Mapika/decider-4b uv run --with transformers==5.17.0 --with flash-linear-attention --with "peft>=0.21" --with torch==2.13.0
       --with torchvision==0.28.0 python scripts/exp_category_mapping.py
"""
import json
import os
import random
import re
from collections import Counter, defaultdict

import numpy as np

from ai_experiments import canon as CN
from ai_experiments import oneslot
from ai_experiments.paths import PROCESSED, ROOT

EMB_MODEL = os.environ.get("EMB_MODEL", "Mapika/decider-4b")
ADAPTER = os.environ.get("ADAPTER", "")
BASELINE = os.environ.get("BASELINE", "") == "1"
CUE = "\nIn one word, the kind of spending:"
INSTRUCT = "Instruct: Given a payee or a budget category, retrieve others that hold the same kind of spending\nQuery:"
READER = "baseline" if BASELINE else ("recipe" if ADAPTER else EMB_MODEL.split("/")[-1])
OPTS = CN.CATEGORY_OPTIONS


def objects():
    out = []
    for it in json.loads((PROCESSED / "kindcat_v2.json").read_text())["items"]:
        name = re.search(r'category "(.*)" hold', it["question"]).group(1)
        hist = it["prompt"].rsplit("\nCategory:", 1)[0]
        filed = []
        for blk in hist.split("\n\n")[1:]:
            t, _, c = blk.partition("\nCategory:")
            if t.startswith("Transaction: ") and c.strip() == name:
                p = oneslot.payee_name(t[len("Transaction: "):])
                if p and p not in filed:
                    filed.append(p)
        out.append(dict(user=it["user"], name=name, gold=it["gold"], level=it["level"], hist=hist,
                        t_name=f'Budget category "{name}"', t_filed=f'Budget category "{name}"' + (f". Payees filed under it: {', '.join(filed[:8])}" if filed else "")))
    return out


def _norm(X):
    return X / np.clip(np.linalg.norm(X, axis=1, keepdims=True), 1e-9, None)


def embed_all(objs):
    """{variant: [n, d]} for the objects and [43, d] for the canonical options."""
    if BASELINE:
        from sklearn.feature_extraction.text import TfidfVectorizer
        v = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5)).fit([o["name"] for o in objs] + OPTS)
        X = _norm(v.transform([o["name"] for o in objs]).toarray()); C = _norm(v.transform(OPTS).toarray())
        return {"name (TF-IDF)": X}, C
    import torch
    from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(EMB_MODEL)
    emb = "Embedding" in EMB_MODEL
    if emb:
        m = AutoModel.from_pretrained(EMB_MODEL, dtype=torch.bfloat16).cuda().eval(); body = m
    else:
        lm = AutoModelForCausalLM.from_pretrained(EMB_MODEL, dtype=torch.bfloat16).cuda().eval()
        if ADAPTER:
            from peft import PeftModel
            lm = PeftModel.from_pretrained(lm, str(ROOT / "models" / "adapters" / ADAPTER)).merge_and_unload().eval()
        body = lm.model
    cue = tok(CUE, add_special_tokens=False)["input_ids"]

    def ids_of(text, prefix=None):
        if emb:
            return tok(INSTRUCT + text, add_special_tokens=False)["input_ids"] + [tok.eos_token_id]
        return (tok(prefix + "\n\n", add_special_tokens=False)["input_ids"] if prefix else []) + tok(text, add_special_tokens=False)["input_ids"] + cue

    @torch.no_grad()
    def run(seqs, bs):
        out = []
        order = sorted(range(len(seqs)), key=lambda i: len(seqs[i]))  # length-sorted batches: the personal texts carry a history
        res = [None] * len(seqs)
        for k in range(0, len(order), bs):
            idx = order[k:k + bs]; ss = [seqs[i] for i in idx]
            T = -(-max(len(s) for s in ss) // 64) * 64
            ids = torch.full((len(ss), T), tok.pad_token_id or 0, dtype=torch.long); att = torch.zeros_like(ids)
            for i, s in enumerate(ss):
                ids[i, :len(s)] = torch.tensor(s); att[i, :len(s)] = 1
            h = body(input_ids=ids.cuda(), attention_mask=att.cuda()).last_hidden_state.float()
            for j, (i, s) in enumerate(zip(idx, ss)):
                res[i] = h[j, len(s) - 1].cpu().numpy()
        return _norm(np.stack(res))

    V = {"name": run([ids_of(o["t_name"]) for o in objs], 64), "name + filed": run([ids_of(o["t_filed"]) for o in objs], 64)}
    if not emb:
        V["personal (history in front)"] = run([ids_of(o["t_name"], o["hist"]) for o in objs], 8)
    C = run([ids_of(f'Budget category "{o}"') for o in OPTS], 64)
    return V, C


def what(o):
    return "several kinds" if o["gold"] == CN.SEVERAL else "person / purpose" if o["gold"] == CN.PURPOSE else "one kind"


if __name__ == "__main__":
    from ai_experiments.licences import open_licence
    if not BASELINE:
        open_licence(EMB_MODEL)
    objs = objects()
    V, C = embed_all(objs)
    gold_idx = np.array([OPTS.index(o["gold"]) for o in objs])
    by_user = defaultdict(list)
    for i, o in enumerate(objs):
        by_user[o["user"]].append(i)
    users = sorted(by_user)
    rng = random.Random(0)
    pairs = [tuple(rng.sample(users, 2)) for _ in range(600)]
    res = {}
    for var, X in V.items():
        S = X @ C.T
        rank = np.array([int(np.where(np.argsort(-S[i]) == gold_idx[i])[0][0]) + 1 for i in range(len(objs))])
        canon = {}
        for g in ["all", "one kind", "several kinds", "person / purpose", "0 rows", "1-2 rows", "3+ rows"]:
            sel = [i for i, o in enumerate(objs) if g == "all" or what(o) == g or o["level"] == g]
            canon[g] = dict(n=len(sel), first=round(100 * float(np.mean(rank[sel] == 1)), 1), top3=round(100 * float(np.mean(rank[sel] <= 3)), 1))
        ok, ok_l = [], defaultdict(list)
        for a, b in pairs:
            B = by_user[b]
            for i in by_user[a]:
                if what(objs[i]) != "one kind" or not any(objs[j]["gold"] == objs[i]["gold"] for j in B):
                    continue
                j = B[int(np.argmax(X[B] @ X[i]))]
                hit = objs[j]["gold"] == objs[i]["gold"]; ok.append(hit)
                ok_l["names differ" if not any(re.sub(r"[^a-z]", "", objs[k]["name"].lower()) == re.sub(r"[^a-z]", "", objs[i]["name"].lower()) for k in B) else "B has the same name"].append(hit)
        uu = dict(n=len(ok), right=round(100 * float(np.mean(ok)), 1), **{k: dict(n=len(v), right=round(100 * float(np.mean(v)), 1)) for k, v in ok_l.items()})
        res[var] = dict(canonical=canon, user_user=uu)
        print(f"{READER} {var}: canonical right first {canon['all']['first']} (top 3 {canon['all']['top3']}; one kind {canon['one kind']['first']}, several "
              f"{canon['several kinds']['first']}, purpose {canon['person / purpose']['first']}); user-user {uu['right']} (names differ "
              f"{uu.get('names differ', {}).get('right')})", flush=True)
    (ROOT / "results" / f"category_mapping_{READER}.json").write_text(json.dumps(dict(reader=READER, model=EMB_MODEL, adapter=ADAPTER, results=res), indent=2))
