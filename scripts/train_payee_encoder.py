"""PLAN step 186 (owner, 2026-10-04: "techniques ... to ensure that two different strings embed closer or further away from each other ...
triples or hard negatives ... with our clean payees and intentionally corrupted payees"): a contrastive payee encoder. A small open
sentence encoder (BASE, default BAAI/bge-small-en-v1.5, MIT) trained with in-batch negatives plus one mined hard negative per pair
(sentence-transformers' MultipleNegativesRankingLoss, i.e. InfoNCE): anchor = a bank string of a merchant (statements.render_v2, the
grammar measured on the owner's strings), positive = another rendering of the same merchant or its plain name, hard negative = a
different merchant that shares the anchor's first word (else a near name by Jaro-Winkler, else any). Merchants: the training split of
realstyle_merchants_v1 (Overture businesses) and of the merchant database's top 48k (build_knowledge_episodes.merchants); the test
split is held out for the check at the end (top-1 retrieval of a rendering among held-out merchants' other renderings).
Person-to-person strings are not trained (their memo, not the person, decides the purpose).
Writes models/encoders/<OUT_NAME>/ (a sentence-transformers model directory; DVC like the adapters).
env: BASE, PAIRS (200000), EPOCHS (1), BATCH (256), LR (5e-5), OUT_NAME (payee_enc_v1), SEED (0).
usage: uv run python scripts/train_payee_encoder.py   (on Modal: modal run scripts/modal_app.py --tag r186-enc --cmd "...")
"""
import json
import os
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from build_knowledge_episodes import merchants  # noqa: E402

from ai_experiments.paths import PROCESSED, ROOT  # noqa: E402
from ai_experiments.statements import render_v2  # noqa: E402

BASE = os.environ.get("BASE", "BAAI/bge-small-en-v1.5")
PAIRS, EPOCHS, BATCH = int(os.environ.get("PAIRS", "200000")), int(os.environ.get("EPOCHS", "1")), int(os.environ.get("BATCH", "256"))
LR, SEED = float(os.environ.get("LR", "5e-5")), int(os.environ.get("SEED", "0"))
OUT = ROOT / "models" / "encoders" / os.environ.get("OUT_NAME", "payee_enc_v1")
NOISE = {"the", "inc", "llc", "co", "sq", "tst", "pos", "sale", "return", "paypal", "debit", "purchase"}


def first_word(name):
    ws = [w for w in re.findall(r"[a-z]+", name.lower()) if w not in NOISE and len(w) > 1]
    return ws[0] if ws else ""


def pool(split):
    rs = json.loads((PROCESSED / "realstyle_merchants_v1.json").read_text())["merchants"]
    out = {m["name"]: m.get("city") for m in rs if m["split"] == split}
    if split == "train":  # the database's popular merchants (it has no city)
        for m in merchants("train", 48000):
            out.setdefault(m["name"], None)
    return [(n, c) for n, c in out.items() if 2 <= len(n) <= 60]


def triplets(ms, n, rng):
    by_first = defaultdict(list)
    for i, (name, _) in enumerate(ms):
        by_first[first_word(name)].append(i)
    out = []
    while len(out) < n:
        i = rng.randrange(len(ms))
        name, city = ms[i]
        a = render_v2(name, rng, city=city)
        p = name if rng.random() < 0.3 else render_v2(name, rng, city=city)
        same = [j for j in by_first.get(first_word(name), []) if j != i]
        j = rng.choice(same) if same and rng.random() < 0.8 else rng.randrange(len(ms))
        if ms[j][0].lower() == name.lower():
            continue
        out.append((a, p, render_v2(ms[j][0], rng, city=ms[j][1]) if rng.random() < 0.7 else ms[j][0]))
    return out


def check(model, rng, k=2000):
    """Held-out merchants: embed two renderings of each of k test-split merchants; top-1 = the other rendering is the nearest."""
    import numpy as np
    ms = rng.sample(pool("test"), k)
    a = [render_v2(n, rng, city=c) for n, c in ms]
    b = [render_v2(n, rng, city=c) for n, c in ms]
    ea, eb = (model.encode(x, normalize_embeddings=True, batch_size=512) for x in (a, b))
    return 100 * float(np.mean((ea @ eb.T).argmax(1) == np.arange(k)))


if __name__ == "__main__":
    from ai_experiments.licences import open_licence
    open_licence(BASE)
    import torch
    from datasets import Dataset
    from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer, SentenceTransformerTrainingArguments, losses
    rng = random.Random(SEED); torch.manual_seed(SEED)
    model = SentenceTransformer(BASE, device="cuda" if torch.cuda.is_available() else "cpu")
    print(f"base {BASE}: held-out top-1 {check(model, random.Random(186)):.1f}", flush=True)
    ms = pool("train")
    tr = triplets(ms, PAIRS, rng)
    print(f"{len(ms)} training merchants, {len(tr)} triplets; e.g. {tr[0]}", flush=True)
    ds = Dataset.from_dict({"anchor": [t[0] for t in tr], "positive": [t[1] for t in tr], "negative": [t[2] for t in tr]})
    args = SentenceTransformerTrainingArguments(output_dir=str(OUT) + "_ckpt", num_train_epochs=EPOCHS, per_device_train_batch_size=BATCH,
                                                learning_rate=LR, warmup_ratio=0.05, bf16=torch.cuda.is_available(), seed=SEED,
                                                batch_sampler="no_duplicates", logging_steps=100, save_strategy="no", report_to=[])
    SentenceTransformerTrainer(model=model, args=args, train_dataset=ds, loss=losses.MultipleNegativesRankingLoss(model)).train()
    print(f"trained: held-out top-1 {check(model, random.Random(186)):.1f}", flush=True)
    model.save(str(OUT))
    print(f"-> {OUT}", flush=True)
