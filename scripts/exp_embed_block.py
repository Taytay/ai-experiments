"""Embedding block (PLAN step 13): the universe and merchant embedding experiments on three encoders, with the
baselines the reviewers asked for, on the SAME frozen items the LLM arms are scored on.

usage: uv run python scripts/exp_embed_block.py ENCODER [PART]     ENCODER in minilm | bge | qwen3; PART in universe | merchant | kge | all
       SMOKE=1 shortens everything (2 epochs, 20 trials, 2 splits) and writes *_smoke files.

Encoders (all through sentence-transformers, so each one's own pooling and normalisation are used):
  minilm   sentence-transformers/all-MiniLM-L6-v2   22M, 384-d, mean pooling, UNCASED WordPiece (the section 4/6 encoder)
  bge      BAAI/bge-base-en-v1.5                     109M, 768-d, CLS pooling, UNCASED WordPiece (the survey called it cased; it lower-cases)
  qwen3    Qwen/Qwen3-Embedding-0.6B                 596M, 1024-d, last-token pooling, CASED Qwen BPE ("Elrholm" and "ELRHOLM" share no token)
  egemma   google/embeddinggemma-300m                308M, 768-d, mean pooling + dense layers, CASED Gemma SentencePiece (added later, gated)
  gtemb    Alibaba-NLP/gte-modernbert-base           149M, 768-d, CLS pooling, ModernBERT BPE (cased), 8k context (added later)

PART universe (MODEL-3, BASE-4): contrastive name -> attribute-text training exactly as exp_universe_embed.py
  (8 epochs, in-batch negatives, same-positive masking), frozen and trained encoder each scored on
    - the frozen ladder induction items (L3_induct_type_{nonsense,realnames,k2,k4}, L4_induct_{weakness,habitat}, L3_induct_heldout):
      the query name's nearest demo name (each demo is one labelled card; the LLM arms answer the same items with the
      same demos in the prompt). Per-item records go to results/per_item/ so the LLM arms can be paired against them.
    - the section 6.3 protocol for continuity (canonical / synonym labels, 300-trial k=1 / k=3 prototypes).
    - the 8-way type task with the five BASE-4 baselines over 10 random splits of the seen species (8 train names per type,
      the other 9 per type held out): frozen centroid, frozen logistic regression, trained centroid, trained logistic
      regression, SetFit (20 same-type pairs per train name, 1,280 pairs, one epoch at 2e-5, then logistic regression) on top of each.
PART merchant (MODEL-3, MODEL-5, BASE-5): exp_embed_vocab.py's conditions on each encoder (zero_shot, ft_subword, ft_newtok_mean)
  plus the BASE-5 new-token variants: ft_newtok_gauss (rows drawn from N(mu, Sigma) of the embedding table), ft_newtok_warm
  (2 epochs on the new rows only, then the full fine-tune), ft_newtok_mosaic (MOSAIC 2510.16797: contrastive + 0.3 x MLM
  on the new-token positions for the first half, contrastive only for the second; tied-embedding encoders only), and for
  the cased encoder ft_newtok_tied (both case forms added, the upper-case row kept equal to the cased row). Metrics: 12-way
  nearest category text for bare name / bank string / description, train and held-out merchants, and a logistic-regression
  head on the name embeddings scored on the bank strings (BASE-4).
PART kge (GRAPH-5): text-initialised inductive KGE: the encoder embeds every entity from its text (merchant name, product,
  city, category text) and a DistMult scorer per relation (has_category, sells, located_in) is trained jointly with the
  encoder on all triples of the training merchants and on the sells / located_in triples of the held-out merchants (their
  category triple is never seen). Held-out-merchant category is then a products-to-category bridge. Against it, on the
  same encoder after the same training: nearest category centroid and logistic regression over the merchant embeddings;
  and a relation-free contrastive control trained on the same triples as (head text, tail text) pairs.
"""
import copy
import json
import os
import random
import sys
import time
import warnings

warnings.filterwarnings("ignore")
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression

from ai_experiments import items as I
from ai_experiments import merchants as M
from ai_experiments import universe as U
from ai_experiments.evals.tracker import Run
from ai_experiments.paths import ROOT

ENCODERS = {"minilm": "sentence-transformers/all-MiniLM-L6-v2", "bge": "BAAI/bge-base-en-v1.5", "qwen3": "Qwen/Qwen3-Embedding-0.6B",
            "egemma": "google/embeddinggemma-300m", "gtemb": "Alibaba-NLP/gte-modernbert-base"}  # both added after section 24 (owner's request; MODEL-3's list)
ENC = sys.argv[1] if len(sys.argv) > 1 else "minilm"
PART = sys.argv[2] if len(sys.argv) > 2 else "all"
MODEL = ENCODERS[ENC]
SMOKE = bool(os.environ.get("SMOKE"))
RUN_TAG = os.environ.get("RUN_TAG", "")
LR = {"minilm": 3e-5, "bge": 3e-5, "qwen3": 1e-5, "egemma": 2e-5, "gtemb": 3e-5}[ENC]
BS, SEED = 32, 0
EPOCHS_U, EPOCHS_M, EPOCHS_K = (2, 2, 2) if SMOKE else (8, 6, 6)
TRIALS, SPLITS = (20, 2) if SMOKE else (300, 10)
OUT = ROOT / "results" / f"embed_block_{ENC}{'_' + RUN_TAG if RUN_TAG else ''}{'_smoke' if SMOKE else ''}.json"
SAVE_DIR = ROOT / ("models/smoke" if SMOKE else "models/adapters")


def save_encoder(model, name):
    """Every trained encoder is kept (owner's rule), in bf16, under models/adapters/embed_<enc>_<name> (DVC)."""
    d = SAVE_DIR / f"embed_{ENC}_{name}"
    model[0].auto_model.to(torch.bfloat16); model.save(str(d)); model[0].auto_model.to(torch.float32)
    print(f"    saved {d.relative_to(ROOT)}", flush=True)
PER_ITEM = ROOT / "results" / "per_item"
torch.manual_seed(SEED)
rng = random.Random(SEED)

# ------------------------------------------------------------------ encoder
from sentence_transformers import SentenceTransformer  # noqa: E402


def load_encoder():
    m = SentenceTransformer(MODEL, device="cuda", model_kwargs={"torch_dtype": torch.float32})  # fp32 master weights; bf16 autocast in forward
    m.max_seq_length = 96
    return m


def tokenize(model, texts):
    fn = getattr(model, "preprocess", None) or model.tokenize
    return {k: v.to("cuda") for k, v in fn(texts).items() if torch.is_tensor(v)}


def embed(model, texts, grad=False, bs=128):
    """Unit-norm sentence embeddings (the model's own pooling + normalisation)."""
    outs = []
    for i in range(0, len(texts), bs if not grad else len(texts)):
        feats = tokenize(model, texts[i:i + (bs if not grad else len(texts))])
        with (torch.enable_grad() if grad else torch.no_grad()), torch.autocast("cuda", dtype=torch.bfloat16):
            e = model(feats)["sentence_embedding"]
        outs.append(F.normalize(e.float(), dim=-1))
    return torch.cat(outs)


def infonce(a, p, chunk):
    scores = a @ p.T * 20.0
    same = torch.tensor([[x[1] == y[1] for y in chunk] for x in chunk], device="cuda")
    scores = scores.masked_fill(same & ~torch.eye(len(chunk), dtype=torch.bool, device="cuda"), -1e4)
    return F.cross_entropy(scores, torch.arange(len(chunk), device="cuda"))


def train_pairs(model, pairs, epochs, lr=LR, params=None, extra_loss=None, log=""):
    """In-batch-negative contrastive training on (anchor, positive) text pairs, same-positive masking (exp_universe_embed)."""
    params = params if params is not None else [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr)
    model.train(); t0 = time.time(); r = random.Random(SEED); loss = torch.tensor(0.0)
    for ep in range(epochs):
        r.shuffle(pairs)
        for i in range(0, len(pairs), BS):
            chunk = pairs[i:i + BS]
            a = embed(model, [p[0] for p in chunk], grad=True); p = embed(model, [p[1] for p in chunk], grad=True)
            loss = infonce(a, p, chunk)
            if extra_loss is not None:
                loss = loss + extra_loss(model, chunk, ep)
            loss.backward(); opt.step(); opt.zero_grad()
    model.eval()
    print(f"    {log}trained {epochs} epochs x {len(pairs)} pairs in {time.time() - t0:.0f}s, final loss {loss.item():.3f}", flush=True)


def acc(pred, gold):
    return round(100 * sum(int(p) == int(g) for p, g in zip(pred, gold)) / len(gold), 1)


def lr_fit(X, y):
    return LogisticRegression(max_iter=3000, C=1.0).fit(X, y)


# ------------------------------------------------------------------ results / tracker
results = {}
per_item_written = []


def save(run, cond, r):
    results[cond] = r
    OUT.parent.mkdir(exist_ok=True); OUT.write_text(json.dumps(results, indent=2))
    run.log({k: v for k, v in r.items() if isinstance(v, (int, float))}, condition=cond)
    print(f"   {cond}: {json.dumps({k: v for k, v in r.items() if isinstance(v, (int, float))})}", flush=True)


def write_per_item(cond, recs):
    p = PER_ITEM / f"embed_block_{ENC}{'_' + RUN_TAG if RUN_TAG else ''}{'_smoke' if SMOKE else ''}.{cond}.jsonl"  # the tag, so a re-run does not overwrite section 24's records
    p.parent.mkdir(exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")
    per_item_written.append(p)


# ================================================================== PART universe
species = U.build()
seen = [s for s in species if not s["heldout"]]
held = [s for s in species if s["heldout"]]
ATTR_TEXT = {"type": {t: U.TYPES[t][0] for t in U.TYPE_LIST},
             "weakness": {t: f"Creatures weak to {t}-type attacks." for t in U.TYPE_LIST},
             "habitat": {h: f"Creatures that live in {h} habitats." for h in U.HABITATS}}
FROZEN = I.load_all(morph=False)
LADDER_LEVELS = ["L3_induct_type_nonsense", "L3_induct_type_realnames", "L3_induct_type_k2", "L3_induct_type_k4",
                 "L4_induct_weakness", "L4_induct_habitat", "L3_induct_heldout"]
LADDER_ITEMS = [it for it in FROZEN.ladder if it["level"] in LADDER_LEVELS]


def universe_pairs():
    pairs = []
    for s in seen:
        for attr in ("type", "weakness", "habitat"):
            pairs += [(s["name"], ATTR_TEXT[attr][s[attr]])] * 2
            pairs += [(f"{s['name']} is a {s['type']}-type creature from {s['region']}.", ATTR_TEXT[attr][s[attr]])]
        pairs += [(s["name"], U.entry(s))] * 2
    return pairs


def score_ladder(model, cond):
    """Every frozen induction item: the query's nearest demo card. Returns per-level accuracy and per-item records."""
    names = sorted({n for it in LADDER_ITEMS for n in it["demos"] + [it["query"]]})
    E = dict(zip(names, embed(model, names)))
    recs, by_level = [], {}
    for it in LADDER_ITEMS:
        sims = torch.stack([E[d] for d in it["demos"]]) @ E[it["query"]]
        pred = int(sims.argmax()); ok = pred == it["answer"]
        recs.append(dict(id=it["id"], level=it["level"], answer=it["answer"], pred=pred, correct=ok, sims=[round(x, 4) for x in sims.tolist()]))
        by_level.setdefault(it["level"], []).append(ok)
    write_per_item(cond, recs)
    return {lvl: round(100 * sum(v) / len(v), 1) for lvl, v in by_level.items()}


def score_universe_protocol(model):
    """Section 6.3's numbers: canonical / synonym labels (8-way) and the 300-trial k=1 / k=3 prototypes."""
    r = {}
    for tag, pool in (("seen", seen), ("heldout", held)):
        names = embed(model, [s["name"] for s in pool])
        for mode, texts in (("canonical", [U.TYPES[t][0] for t in U.TYPE_LIST]), ("synonym", [U.TYPES[t][1][0] for t in U.TYPE_LIST])):
            labs = embed(model, texts)
            r[f"type_{mode}_{tag}"] = acc((names @ labs.T).argmax(1).tolist(), [U.TYPE_LIST.index(s["type"]) for s in pool])
        E = dict(zip([s["name"] for s in pool], names))
        for attr in ("type", "weakness", "habitat"):
            for k in (1, 3):
                counts = {v: sum(s[attr] == v for s in pool) for v in {s[attr] for s in pool}}
                usable = sorted(v for v, c in counts.items() if c > k)
                if len(usable) < 3:
                    continue
                hits, tr = [], random.Random(SEED + k)
                for _ in range(TRIALS):
                    vals = tr.sample(usable, 3)
                    demos = {v: tr.sample([s for s in pool if s[attr] == v], k) for v in vals}
                    protos = F.normalize(torch.stack([torch.stack([E[d["name"]] for d in demos[v]]).mean(0) for v in vals]), dim=-1)
                    qv = tr.choice(vals)
                    q = tr.choice([s for s in pool if s[attr] == qv and s not in demos[qv]])
                    hits.append(vals[int((E[q["name"]] @ protos.T).argmax())] == qv)
                r[f"proto_{attr}_k{k}_{tag}"] = round(100 * sum(hits) / len(hits), 1)
    return r


def type_task_baselines(model, cond):
    """8-way type of seen species from the name embedding, 10 random splits (8 train names per type): centroid, logistic
    regression, and SetFit (label-contrastive pairs on the train names, one epoch, then logistic regression)."""
    out = {"centroid": [], "logreg": [], "setfit": []}
    state = copy.deepcopy(model.state_dict())
    for split in range(SPLITS):
        sr = random.Random(100 + split)
        train, test = [], []
        for t in U.TYPE_LIST:
            pool = [s for s in seen if s["type"] == t]; sr.shuffle(pool)
            train += pool[:8]; test += pool[8:]
        ytr = [U.TYPE_LIST.index(s["type"]) for s in train]; yte = [U.TYPE_LIST.index(s["type"]) for s in test]
        Xtr = embed(model, [s["name"] for s in train]); Xte = embed(model, [s["name"] for s in test])
        cents = F.normalize(torch.stack([Xtr[[i for i, y in enumerate(ytr) if y == c]].mean(0) for c in range(len(U.TYPE_LIST))]), dim=-1)
        out["centroid"].append(acc((Xte @ cents.T).argmax(1).tolist(), yte))
        out["logreg"].append(acc(lr_fit(Xtr.cpu().numpy(), ytr).predict(Xte.cpu().numpy()), yte))
        # SetFit: R=20 positive pairs per TRAIN NAME (same type = positive; 64 x 20 = 1,280 pairs), in-batch negatives, 1 epoch
        pairs = []
        for s_, c in zip(train, ytr):
            others = [x["name"] for x, y in zip(train, ytr) if y == c and x is not s_]
            for _ in range(20):
                pairs.append((s_["name"], str(c) + ":" + sr.choice(others)))
        # positives are (class, name) strings; the same-positive mask must treat same-class pairs as non-negatives
        chunks_fn = lambda chunk: [(x[0], x[1].split(":")[0]) for x in chunk]
        opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=2e-5)
        model.train(); sr.shuffle(pairs)
        for i in range(0, len(pairs), BS):
            chunk = pairs[i:i + BS]
            a = embed(model, [p[0] for p in chunk], grad=True); p = embed(model, [p[1].split(":", 1)[1] for p in chunk], grad=True)
            loss = infonce(a, p, chunks_fn(chunk)); loss.backward(); opt.step(); opt.zero_grad()
        model.eval()
        Xtr2 = embed(model, [s["name"] for s in train]); Xte2 = embed(model, [s["name"] for s in test])
        out["setfit"].append(acc(lr_fit(Xtr2.cpu().numpy(), ytr).predict(Xte2.cpu().numpy()), yte))
        model.load_state_dict(state)
    r = {}
    for k, v in out.items():
        r[f"type8_{k}_mean"] = round(float(np.mean(v)), 1); r[f"type8_{k}_sd"] = round(float(np.std(v, ddof=1)) if len(v) > 1 else 0.0, 1)
    return r


def part_universe(run):
    model = load_encoder()
    for cond in ("frozen", "trained"):
        if cond == "trained":
            train_pairs(model, universe_pairs(), EPOCHS_U, log="universe: ")
            save_encoder(model, "universe_trained")
        r = {}
        r.update({f"ladder_{k}": v for k, v in score_ladder(model, f"universe_{cond}").items()})
        r.update(score_universe_protocol(model))
        r.update(type_task_baselines(model, cond))
        save(run, f"universe_{cond}", r)
    del model; torch.cuda.empty_cache()


# ================================================================== PART merchant
all_m = M.build()
mrng = random.Random(SEED)
held_names = {m["name"] for c in M.CATEGORY_LIST for m in mrng.sample([x for x in all_m if x["category"] == c], 2)}
train_m = [m for m in all_m if m["name"] not in held_names]
held_m = [m for m in all_m if m["name"] in held_names]
CAT_TEXT = {c: f"{c}: {', '.join(M.CATEGORIES[c])}" for c in M.CATEGORY_LIST}


def merchant_pairs():
    pairs = []
    for m in train_m:
        pos = CAT_TEXT[m["category"]]
        pairs += [(t, pos) for t in M.augmented(m)]
        pairs += [(m["name"], pos)] * 3
    return pairs


def score_merchant(model):
    cats = embed(model, [CAT_TEXT[c] for c in M.CATEGORY_LIST])
    r = {}
    for tag, ms in (("train", train_m), ("heldout", held_m)):
        labels = [M.CATEGORY_LIST.index(m["category"]) for m in ms]
        for fmt, texts in (("name", [m["name"] for m in ms]), ("bank", [M.bank_string(m) for m in ms]), ("desc", [M.raw_fact(m) for m in ms])):
            r[f"{fmt}_{tag}"] = acc((embed(model, texts) @ cats.T).argmax(1).tolist(), labels)
    # BASE-4: logistic regression on the training merchants' name + sentence embeddings, scored on the bank strings
    Xtr = embed(model, [m["name"] for m in train_m] + [t for m in train_m for t in M.augmented(m)])
    ytr = [M.CATEGORY_LIST.index(m["category"]) for m in train_m] + [M.CATEGORY_LIST.index(m["category"]) for m in train_m for _ in M.augmented(m)]
    clf = lr_fit(Xtr.cpu().numpy(), ytr)
    for tag, ms in (("train", train_m), ("heldout", held_m)):
        labels = [M.CATEGORY_LIST.index(m["category"]) for m in ms]
        r[f"logreg_bank_{tag}"] = acc(clf.predict(embed(model, [M.bank_string(m) for m in ms]).cpu().numpy()), labels)
        r[f"logreg_name_{tag}"] = acc(clf.predict(embed(model, [m["name"] for m in ms]).cpu().numpy()), labels)
    return r


def add_tokens(model, names, init, tie_upper=False):
    """Add one token per merchant name (both case forms when tie_upper), initialised from the subword rows (mean), from
    N(mu, Sigma) of the table (gauss), or at random (random). Returns the new token ids (cased, upper)."""
    tok = model.tokenizer; tr = model[0].auto_model
    emb = tr.get_input_embeddings().weight
    cased = getattr(tok, "do_lower_case", None) is not True
    forms = [n if cased else n.lower() for n in names]
    with torch.no_grad():
        means = [emb[tok(n, add_special_tokens=False)["input_ids"]].mean(0).clone() for n in names]
        mu, std = emb.mean(0), emb.std().item()
        if init == "gauss":
            X = emb.float() - mu.float()
            cov = X.T @ X / (X.shape[0] - 1) + 1e-6 * torch.eye(X.shape[1], device=emb.device)
            dist = torch.distributions.MultivariateNormal(mu.float(), covariance_matrix=cov)
    tok.add_tokens(forms + ([n.upper() for n in names] if tie_upper else []))
    tr.resize_token_embeddings(len(tok), mean_resizing=False)
    emb = tr.get_input_embeddings().weight
    ids, upper_ids = [], []
    with torch.no_grad():
        for n, f, v in zip(names, forms, means):
            i = tok.convert_tokens_to_ids(f); ids.append(i)
            emb[i] = v if init == "mean" else (dist.sample().to(emb.dtype) if init == "gauss" else torch.randn_like(v) * std)
            if tie_upper:
                j = tok.convert_tokens_to_ids(n.upper()); upper_ids.append(j); emb[j] = emb[i]
    return ids, upper_ids


def mosaic_mlm(model, new_ids, alpha=0.3, p_mask=0.15, mask_id=None):
    """MOSAIC's joint stage: MLM restricted to the new-token positions, logits through the tied input embedding table."""
    tr = model[0].auto_model; new = torch.tensor(new_ids, device="cuda")

    def loss_fn(model, chunk, ep):
        if ep >= EPOCHS_M // 2:
            return torch.tensor(0.0, device="cuda")
        feats = tokenize(model, [c[0] for c in chunk]); ids = feats["input_ids"]
        is_new = torch.isin(ids, new)
        mask = is_new & (torch.rand_like(ids, dtype=torch.float) < p_mask)
        if not mask.any():
            return torch.tensor(0.0, device="cuda")
        inp = ids.clone(); inp[mask] = mask_id
        with torch.autocast("cuda", dtype=torch.bfloat16):
            h = tr(input_ids=inp, attention_mask=feats["attention_mask"]).last_hidden_state[mask]
            logits = h.float() @ tr.get_input_embeddings().weight.float().T
        return alpha * F.cross_entropy(logits, ids[mask])
    return loss_fn


def part_merchant(run):
    names = [m["name"] for m in all_m]
    conds = ["zero_shot", "ft_subword", "ft_newtok_mean", "ft_newtok_gauss", "ft_newtok_warm"] + (["ft_newtok_mosaic"] if ENC in ("minilm", "bge") else ["ft_newtok_tied"])
    for cond in conds:
        model = load_encoder()
        if cond != "zero_shot":
            pairs = merchant_pairs(); extra = None; params = None
            if cond.startswith("ft_newtok"):
                init = "gauss" if cond == "ft_newtok_gauss" else "mean"
                ids, upper = add_tokens(model, names, init, tie_upper=(cond == "ft_newtok_tied"))
                print(f"    {cond}: added {len(ids) + len(upper)} tokens ({init} init)", flush=True)
                emb = model[0].auto_model.get_input_embeddings().weight
                if cond == "ft_newtok_warm":  # new rows only, then everything
                    keep = torch.zeros(emb.shape[0], 1, device="cuda"); keep[ids] = 1.0
                    h = emb.register_hook(lambda g: g * keep.to(g.dtype))
                    for p in model.parameters(): p.requires_grad_(False)
                    emb.requires_grad_(True)
                    train_pairs(model, pairs, 2, lr=1e-3, params=[emb], log="warm-up (new rows only): ")
                    h.remove()
                    for p in model.parameters(): p.requires_grad_(True)
                if cond == "ft_newtok_mosaic":
                    extra = mosaic_mlm(model, ids, mask_id=model.tokenizer.mask_token_id)
                if cond == "ft_newtok_tied":
                    tie = (ids, upper)
                    def extra(model, chunk, ep, tie=tie):  # keep the upper-case rows equal to the cased rows
                        w = model[0].auto_model.get_input_embeddings().weight
                        with torch.no_grad(): w[tie[1]] = w[tie[0]]
                        return torch.tensor(0.0, device="cuda")
            train_pairs(model, pairs, EPOCHS_M, extra_loss=extra, log=f"{cond}: ")
            if cond == "ft_newtok_tied":
                w = model[0].auto_model.get_input_embeddings().weight
                with torch.no_grad(): w[upper] = w[ids]
        r = score_merchant(model)
        if cond != "zero_shot":
            save_encoder(model, f"merchant_{cond}")
        if cond.startswith("ft_newtok"):
            tok = model.tokenizer
            r["bank_hits_new_token"] = round(100 * sum(any(i in ids + (upper if cond == "ft_newtok_tied" else []) for i in tok(M.bank_string(m), add_special_tokens=False)["input_ids"]) for m in all_m) / len(all_m), 1)
        save(run, f"merchant_{cond}", r)
        del model; torch.cuda.empty_cache()


# ================================================================== PART kge (GRAPH-5)
def kge_triples():
    """(head text, relation, tail text). Training merchants: all three relations; held-out merchants: sells and located_in only."""
    tr = []
    for m in all_m:
        for p in m["products"]:
            tr.append((m["name"], "sells", p))
        tr.append((m["name"], "located_in", m["city"]))
        if m["name"] not in held_names:
            tr.append((m["name"], "has_category", CAT_TEXT[m["category"]]))
    return tr


def part_kge(run):
    triples = kge_triples()
    tails = {r: sorted({t for _, rr, t in triples if rr == r}) for r in ("sells", "located_in", "has_category")}
    print(f"    kge: {len(triples)} triples; tails sells {len(tails['sells'])}, located_in {len(tails['located_in'])}, categories {len(tails['has_category'])}", flush=True)

    def score_after(model, rel=None):
        """Held-out and training merchant category from name and from bank string: the relation scorer (if given), the
        nearest category centroid of the training merchants' name embeddings, logistic regression on the same, and the
        nearest category text."""
        r = {}
        Xtr = embed(model, [m["name"] for m in train_m]); ytr = [M.CATEGORY_LIST.index(m["category"]) for m in train_m]
        cents = F.normalize(torch.stack([Xtr[[i for i, y in enumerate(ytr) if y == c]].mean(0) for c in range(len(M.CATEGORY_LIST))]), dim=-1)
        clf = lr_fit(Xtr.cpu().numpy(), ytr)
        cats = embed(model, [CAT_TEXT[c] for c in M.CATEGORY_LIST])
        for tag, ms in (("train", train_m), ("heldout", held_m)):
            labels = [M.CATEGORY_LIST.index(m["category"]) for m in ms]
            for fmt, texts in (("name", [m["name"] for m in ms]), ("bank", [M.bank_string(m) for m in ms])):
                X = embed(model, texts)
                r[f"centroid_{fmt}_{tag}"] = acc((X @ cents.T).argmax(1).tolist(), labels)
                r[f"logreg_{fmt}_{tag}"] = acc(clf.predict(X.cpu().numpy()), labels)
                r[f"cattext_{fmt}_{tag}"] = acc((X @ cats.T).argmax(1).tolist(), labels)
                if rel is not None:
                    r[f"kge_{fmt}_{tag}"] = acc(((X * rel["has_category"]) @ cats.T).argmax(1).tolist(), labels)
        return r

    # (a) the KGE: DistMult relation vectors, tails of each relation as the candidate set, encoder trained jointly
    model = load_encoder(); d = model.get_sentence_embedding_dimension()
    rel = torch.nn.ParameterDict({r: torch.nn.Parameter(torch.ones(d, device="cuda")) for r in tails})
    opt = torch.optim.AdamW([{"params": [p for p in model.parameters() if p.requires_grad], "lr": LR}, {"params": list(rel.parameters()), "lr": 1e-2}])
    model.train(); t0 = time.time(); kr = random.Random(SEED)
    for ep in range(EPOCHS_K):
        kr.shuffle(triples)
        for i in range(0, len(triples), BS):
            chunk = triples[i:i + BS]
            h = embed(model, [c[0] for c in chunk], grad=True)
            loss = 0.0
            for rname in tails:
                idx = [j for j, c in enumerate(chunk) if c[1] == rname]
                if not idx:
                    continue
                T = embed(model, tails[rname], grad=True)
                scores = (h[idx] * rel[rname]) @ T.T * 20.0
                gold = torch.tensor([tails[rname].index(chunk[j][2]) for j in idx], device="cuda")
                loss = loss + F.cross_entropy(scores, gold) * len(idx) / len(chunk)
            loss.backward(); opt.step(); opt.zero_grad()
    model.eval()
    print(f"    kge: trained {EPOCHS_K} epochs in {time.time() - t0:.0f}s, final loss {float(loss):.3f}", flush=True)
    save(run, "kge_distmult", score_after(model, {k: v.detach() for k, v in rel.items()}))
    save_encoder(model, "kge_distmult"); torch.save({k: v.detach().cpu() for k, v in rel.items()}, SAVE_DIR / f"embed_{ENC}_kge_distmult" / "relations.pt")
    del model; torch.cuda.empty_cache()
    # (b) relation-free control: the same triples as (head text, tail text) contrastive pairs
    model = load_encoder()
    train_pairs(model, [(h, t) for h, _, t in triples], EPOCHS_K, log="kge control (pairs): ")
    save(run, "kge_pairs_control", score_after(model))
    save_encoder(model, "kge_pairs_control")
    del model; torch.cuda.empty_cache()


# ================================================================== main
cfg = dict(encoder=ENC, part=PART, run_tag=RUN_TAG, lr=LR, bs=BS, seed=SEED, epochs_universe=EPOCHS_U, epochs_merchant=EPOCHS_M, epochs_kge=EPOCHS_K,
           trials=TRIALS, splits=SPLITS, smoke=SMOKE, n_species=len(species), n_heldout_species=len(held), n_merchants=len(all_m),
           n_heldout_merchants=len(held_m), **FROZEN.config())
with Run("embed_block", model=MODEL, config=cfg, enabled=not SMOKE) as run:
    if PART in ("universe", "all"):
        print("== universe", flush=True); part_universe(run)
    if PART in ("merchant", "all"):
        print("== merchant", flush=True); part_merchant(run)
    if PART in ("kge", "all"):
        print("== kge", flush=True); part_kge(run)
    if not SMOKE:
        run.artifact(OUT)
        for p in per_item_written:
            run.artifact(p)
print(f"\nwrote {OUT.relative_to(ROOT)}")
