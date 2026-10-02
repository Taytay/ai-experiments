"""PLAN step 157 (owner, 2026-10-02: "We can extract embeddings from decider somehow, right? Let's experiment with doing so, and with
clustering to see if it works."). Embeddings from decoder hidden states, judged against the generators' own labels (blind_v2):
  payees      kindpay_v2: each payee rendered three ways (statement name only; name + other users' filings; filings only), gold = its
              canonical kind (ai_experiments.canon, 41 kinds)
  categories  kindcat_v2: each (user, category): 'Budget category "<name>"', plus the payees the user filed under it in the history slice
              (up to 8, oneslot.payee_name), gold = what it holds (a canonical kind, "several kinds", or a person / trip / purpose)
Readers (EMB_MODEL): a decoder (Mapika/decider-4b; ADAPTER = a LoRA from exp_decider_finetune.py, merged; Qwen/Qwen3.5-4B-Base),
read as the hidden state at the last token of the text followed by a cue ("...\\nIn one word, the kind of spending:"; PromptEOL style)
and as the mean over the text's tokens, at 1/2, 3/4 and all of the depth; or Qwen/Qwen3-Embedding-4B (Apache-2.0), a purpose-built
embedder, read its documented way (an instruction, last-token pooling, final layer). No-model baselines (BASELINE=1, CPU): character
n-gram TF-IDF of the payee text; TF-IDF of the filed payee names for categories.
Objects deduplicated by (level, text). Measures per object set and level (payees only per rendering): leave-one-out 1-NN and 10-NN majority accuracy (cosine) against the gold kind (for categories also the 1-NN among categories with
another name); k-means with k = the
number of gold kinds: NMI, ARI, purity. Writes results/embed_cluster_<reader>.json.
usage: EMB_MODEL=Mapika/decider-4b uv run --with transformers==5.17.0 --with flash-linear-attention --with "peft>=0.21" --with torch==2.13.0
       --with torchvision==0.28.0 python scripts/exp_embed_cluster.py
"""
import json
import os
import re
from collections import Counter, defaultdict

import numpy as np

from ai_experiments import oneslot
from ai_experiments.paths import PROCESSED, ROOT

EMB_MODEL = os.environ.get("EMB_MODEL", "Mapika/decider-4b")
ADAPTER = os.environ.get("ADAPTER", "")
BASELINE = os.environ.get("BASELINE", "") == "1"
BATCH = int(os.environ.get("BATCH", "32"))
CUE = "\nIn one word, the kind of spending:"
INSTRUCT = "Instruct: Given a payee or a budget category, retrieve others that hold the same kind of spending\nQuery:"
READER = "baseline" if BASELINE else (ADAPTER and "recipe-" + ADAPTER[:40] or EMB_MODEL.split("/")[-1])


def payees():
    out = []
    for it in json.loads((PROCESSED / "kindpay_v2.json").read_text())["items"]:
        text = it["prompt"].rsplit("\nCategory:", 1)[0].strip()
        out.append(dict(id=it["id"], text=text, gold=it["gold"], level=it["level"].split(" / ")[0]))
    return out


def categories():
    out = []
    for it in json.loads((PROCESSED / "kindcat_v2.json").read_text())["items"]:
        name = re.search(r'category "(.*)" hold', it["question"]).group(1)
        filed = []
        for blk in it["prompt"].rsplit("\nCategory:", 1)[0].split("\n\n")[1:]:
            t, _, c = blk.partition("\nCategory:")
            if t.startswith("Transaction: ") and c.strip() == name:
                p = oneslot.payee_name(t[len("Transaction: "):])
                if p and p not in filed:
                    filed.append(p)
        text = f'Budget category "{name}"' + (f". Payees filed under it: {', '.join(filed[:8])}" if filed else "")
        out.append(dict(id=it["id"], text=text, gold=it["gold"], level=it["level"], filed=filed[:8], group=re.sub(r"[^a-z]", "", name.lower())))
    return out


def dedupe(objs):
    """One object per distinct (level, text), labelled with the commonest gold among its copies: identical texts would otherwise be each
    other's nearest neighbours (many users share a bare category name such as "Eating Out")."""
    groups = defaultdict(list)
    for o in objs:
        groups[(o["level"], o["text"])].append(o)
    return [dict(g[0], gold=Counter(o["gold"] for o in g).most_common(1)[0][0], copies=len(g)) for g in groups.values()]


def decoder_embed(texts):
    """{(pool, layer fraction): [n, d] float32, L2-normalised}."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(EMB_MODEL)
    lm = AutoModelForCausalLM.from_pretrained(EMB_MODEL, dtype=torch.bfloat16).cuda().eval()
    if ADAPTER:
        from peft import PeftModel
        lm = PeftModel.from_pretrained(lm, str(ROOT / "models" / "adapters" / ADAPTER)).merge_and_unload().eval()
    out = defaultdict(list)
    with torch.no_grad():
        for k in range(0, len(texts), BATCH):
            chunk = texts[k:k + BATCH]
            enc = [tok(t, add_special_tokens=False)["input_ids"] for t in chunk]
            cue = tok(CUE, add_special_tokens=False)["input_ids"]
            seqs = [e + cue for e in enc]
            T = -(-max(len(s) for s in seqs) // 64) * 64
            ids = torch.full((len(seqs), T), tok.pad_token_id or 0, dtype=torch.long); att = torch.zeros_like(ids)
            for i, s in enumerate(seqs):
                ids[i, :len(s)] = torch.tensor(s); att[i, :len(s)] = 1
            hs = lm.model(input_ids=ids.cuda(), attention_mask=att.cuda(), output_hidden_states=True).hidden_states
            n = len(hs) - 1
            for frac in (0.5, 0.75, 1.0):
                h = hs[round(frac * n)].float()
                for i, (e, s) in enumerate(zip(enc, seqs)):
                    out[("last", frac)].append(h[i, len(s) - 1].cpu().numpy())
                    out[("mean", frac)].append(h[i, :len(e)].mean(0).cpu().numpy())
    return {k: _norm(np.stack(v)) for k, v in out.items()}


def embedder_embed(texts):
    import torch
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(EMB_MODEL)
    m = AutoModel.from_pretrained(EMB_MODEL, dtype=torch.bfloat16).cuda().eval()
    vecs = []
    with torch.no_grad():
        for k in range(0, len(texts), BATCH):
            seqs = [tok(INSTRUCT + t, add_special_tokens=False)["input_ids"] + [tok.eos_token_id] for t in texts[k:k + BATCH]]
            T = max(len(s) for s in seqs)
            ids = torch.full((len(seqs), T), tok.pad_token_id or 0, dtype=torch.long); att = torch.zeros_like(ids)
            for i, s in enumerate(seqs):
                ids[i, :len(s)] = torch.tensor(s); att[i, :len(s)] = 1
            h = m(input_ids=ids.cuda(), attention_mask=att.cuda()).last_hidden_state.float()
            vecs += [h[i, len(s) - 1].cpu().numpy() for i, s in enumerate(seqs)]
    return {("last", 1.0): _norm(np.stack(vecs))}


def baseline_embed(objs, kind):
    from sklearn.feature_extraction.text import TfidfVectorizer
    if kind == "payees":
        docs = [o["text"] for o in objs]
        X = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2).fit_transform(docs)
    else:
        docs = [" ".join(p.lower().replace(" ", "_") for p in o["filed"]) + " " + re.sub(r"[^a-z ]", "", o["text"].split('"')[1].lower()) for o in objs]
        X = TfidfVectorizer(min_df=2).fit_transform(docs)
    from sklearn.decomposition import TruncatedSVD
    Z = TruncatedSVD(min(128, X.shape[1] - 1), random_state=0).fit_transform(X)
    return {("tfidf-svd", 1.0): _norm(Z)}


def _norm(X):
    return X / np.clip(np.linalg.norm(X, axis=1, keepdims=True), 1e-9, None)


def measures(X, gold, groups=None):
    from sklearn.cluster import KMeans
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
    y = np.array(gold); S = X @ X.T; np.fill_diagonal(S, -np.inf)
    nn = np.argsort(-S, 1)[:, :10]
    knn1 = float(np.mean(y[nn[:, 0]] == y))
    knn10 = float(np.mean([Counter(y[r]).most_common(1)[0][0] == t for r, t in zip(nn, y)]))
    extra = {}
    if groups is not None:  # the nearest neighbour with another name: meaning, not name matching
        g = np.array(groups); S2 = S.copy(); S2[g[:, None] == g[None, :]] = -np.inf
        extra["knn1_othername"] = round(100 * float(np.mean(y[np.argmax(S2, 1)] == y)), 1)
    k = len(set(gold))
    lab = KMeans(k, n_init=4, random_state=0).fit_predict(X)
    pur = sum(Counter(y[lab == c]).most_common(1)[0][1] for c in set(lab)) / len(y)
    return dict(n=len(y), k=k, knn1=round(100 * knn1, 1), knn10=round(100 * knn10, 1), nmi=round(normalized_mutual_info_score(y, lab), 3),
                ari=round(adjusted_rand_score(y, lab), 3), purity=round(100 * pur, 1), **extra), lab


if __name__ == "__main__":
    from ai_experiments.licences import open_licence
    if not BASELINE:
        open_licence(EMB_MODEL)
    res = {}
    for kind, objs in (("payees", dedupe(payees())), ("categories", dedupe(categories()))):
        texts = [o["text"] for o in objs]
        embs = baseline_embed(objs, kind) if BASELINE else embedder_embed(texts) if "Embedding" in EMB_MODEL else decoder_embed(texts)
        levels = sorted({o["level"] for o in objs})
        for (pool, frac), X in embs.items():
            key = f"{kind} | {pool} @ {frac:g}"
            res[key] = {}
            for lv in (levels if kind == "payees" else ["all"] + levels):  # a payee appears once per rendering: never pool renderings
                sel = [i for i, o in enumerate(objs) if lv == "all" or o["level"] == lv]
                if len(sel) >= 50:
                    m, lab = measures(X[sel], [objs[i]["gold"] for i in sel], [objs[i]["group"] for i in sel] if kind == "categories" else None)
                    res[key][lv] = m
            print(f"{READER} {key}: " + " | ".join(f"{lv} 1-NN {m['knn1']}" + (f" (other name {m['knn1_othername']})" if "knn1_othername" in m else "") + f" 10-NN {m['knn10']} NMI {m['nmi']} purity {m['purity']}" for lv, m in res[key].items()), flush=True)
        # the clusters of one configuration, named by their commonest gold label, for a look
        pick = ("last", 0.75) if ("last", 0.75) in embs else next(iter(embs))
        _, lab = measures(embs[pick], [o["gold"] for o in objs])
        for c in sorted(set(lab), key=lambda c: -np.sum(lab == c))[:8]:
            mem = [objs[i] for i in np.where(lab == c)[0]]
            top = Counter(o["gold"] for o in mem).most_common(1)[0]
            print(f"   cluster {c} ({len(mem)}; {top[1]} '{top[0]}'): " + "; ".join(o["text"].split("\n")[0].split(": ", 1)[-1][:40] for o in mem[:6]), flush=True)
    (ROOT / "results" / f"embed_cluster_{READER}.json").write_text(json.dumps(dict(reader=READER, model=EMB_MODEL, adapter=ADAPTER, results=res), indent=2))
