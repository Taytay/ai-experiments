"""PLAN step 187 (owner, 2026-10-04: "Could we push categories and payees together embeddings wise using this same mechanism?"): a
two-tower (dual) encoder. A transaction ("<bank string> | $<amount> | <weekday>") and a category ("<Group>: <Name>; recently: <the three
payees last filed under it>", or "; nothing filed yet") are embedded by one small open encoder (BASE, default BAAI/bge-small-en-v1.5,
MIT) and pulled together with InfoNCE (sentence-transformers' MultipleNegativesRankingLoss): in-batch negatives plus one hard negative,
another category of the same household as it stood that day. Categories are described by what is filed in them because their names
alone are personal (REPORT 154). Only filings before the transaction's date describe a category, in training and in reading.
Training: shared-world synthetic households (realstyle, SHARED_WORLD=1, train split). Reading: a budget replayed in date order, every
visible category scored by cosine; top-1 / top-3 / top-10 and whether the payee was filed before. The owner's budget is read on this
machine only (real_budget_eval.py's CACHE), printing aggregates.
  train  writes models/encoders/<OUT_NAME>/ (DVC)
  read   READ=households (TEST_SEEDS of the test split, a new world) or READ=budget (BUDGET, as real_budget_eval.py)
env: BASE, PAIRS (200000), HOUSEHOLDS (400), BATCH (128), EPOCHS (1), OUT_NAME (two_tower_v1), ENC (for read: an encoders/ name, or BASE
     untrained), READ, TEST_SEEDS ("100000-100049"), SEED (0).
usage: SHARED_WORLD=1 GROUPNAMES=1 uv run python scripts/two_tower.py train|read
"""
import datetime as dt
import json
import os
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from ai_experiments.paths import ROOT  # noqa: E402

BASE = os.environ.get("BASE", "BAAI/bge-small-en-v1.5")
PAIRS, HOUSEHOLDS = int(os.environ.get("PAIRS", "200000")), int(os.environ.get("HOUSEHOLDS", "400"))
BATCH, EPOCHS, SEED = int(os.environ.get("BATCH", "128")), int(os.environ.get("EPOCHS", "1")), int(os.environ.get("SEED", "0"))
OUT = ROOT / "models" / "encoders" / os.environ.get("OUT_NAME", "two_tower_v1")
WD = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def _clean(s):
    return re.sub(r"\s+", " ", (s or "")).strip()


def replay(b):
    """[(transaction text, gold category id, {category id: category text as of that day}, payee seen before)] for a YNAB budget document,
    in date order; inflows, transfers, splits and unapproved or uncategorised transactions left out."""
    groups = {g["id"]: g["name"] for g in b["category_groups"]}
    cats = {c["id"]: c for c in b["categories"] if not c.get("deleted")}
    payees = {p["id"]: p["name"] for p in b["payees"]}
    split_parents = {s["transaction_id"] for s in b.get("subtransactions", []) if not s.get("deleted")}
    label = {cid: f'{_clean(groups.get(c["category_group_id"], ""))}: {_clean(c["name"])}' for cid, c in cats.items()
             if groups.get(c["category_group_id"]) != "Internal Master Category"}
    visible = [cid for cid in label if not cats[cid].get("hidden")]
    txs = sorted((t for t in b["transactions"] if not t.get("deleted") and t.get("approved") and t.get("category_id") in label
                  and not t.get("transfer_account_id") and t["id"] not in split_parents and t["amount"] < 0), key=lambda t: (t["date"], t["id"]))
    recent, seen, out = defaultdict(list), set(), []
    i = 0
    while i < len(txs):
        day = txs[i]["date"]
        j = i
        while j < len(txs) and txs[j]["date"] == day:
            j += 1
        state = {cid: label[cid] + ("; recently: " + ", ".join(recent[cid][:3]) if recent[cid] else "; nothing filed yet") for cid in visible}
        for t in txs[i:j]:
            p = _clean(payees.get(t.get("payee_id"), "") or t.get("import_payee_name_original") or "")
            d = dt.date.fromisoformat(day)
            out.append((f"{p} | ${-t['amount'] / 1000:.2f} | {WD[d.weekday()]}", t["category_id"], state, p in seen))
        for t in txs[i:j]:  # the day's filings describe the categories from the next day on
            p = _clean(payees.get(t.get("payee_id"), "") or "")
            recent[t["category_id"]] = [p] + [x for x in recent[t["category_id"]] if x != p][:4]
            seen.add(p)
        i = j
    return out


def _hh_key(split):
    """the generator's identity: realstyle.py and statements.py source, the data files they read, and the env flags that change worlds"""
    import hashlib
    from ai_experiments import realstyle as R, statements as S
    from ai_experiments.paths import PROCESSED
    h = hashlib.sha1()
    for f in (R.__file__, S.__file__):
        h.update(Path(f).read_bytes())
    for f in sorted(PROCESSED.glob("statement_patterns_v*.json")) + [PROCESSED / "realstyle_merchants_v1.json", PROCESSED / "category_style_v1.json"]:
        if f.exists():
            h.update(f"{f.name}{f.stat().st_size}{f.stat().st_mtime_ns}".encode())
    h.update(f"{os.environ.get('SHARED_WORLD')}|{os.environ.get('REALSTYLE_V4')}|{split}".encode())
    if os.environ.get("RS_V5") == "1":  # row 225 (keys of other settings unchanged)
        h.update(b"|v5")
        if os.environ.get("RS_V6"):  # row 233
            h.update(f"|v6{''.join(sorted(os.environ['RS_V6']))}|{os.environ.get('RS_V6A', '0.25')}|{os.environ.get('RS_V6M', '0.3')}".encode())
    return h.hexdigest()[:16]


def households(split, seeds):
    """synthetic budgets; cached on disk (data/interim/hh_cache, gitignored; HH_CACHE=0 turns it off): building one takes ~0.4 s, loading
    it ~0.02 s (2026-10-05: every 100-household read was rebuilding the same households). The key covers the generator's code, data and
    env flags, so a change to any of them builds afresh."""
    import pickle
    from ai_experiments import realstyle as R
    from ai_experiments.paths import ROOT
    cache = None
    if os.environ.get("HH_CACHE", "1") != "0":
        cache = ROOT / "data" / "interim" / "hh_cache" / _hh_key(split)
        cache.mkdir(parents=True, exist_ok=True)
    for s in seeds:
        f = cache / f"{split}_{s}.pkl" if cache else None
        if f and f.exists():
            yield pickle.loads(f.read_bytes())
            continue
        b = R.household(s, split).budget()["budget"]
        if f:
            tmp = f.with_suffix(f".tmp{os.getpid()}")
            tmp.write_bytes(pickle.dumps(b, protocol=5))
            tmp.replace(f)
        yield b


def train():
    from ai_experiments.licences import open_licence
    open_licence(BASE)
    import torch
    from datasets import Dataset
    from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer, SentenceTransformerTrainingArguments, losses
    rng = random.Random(SEED); torch.manual_seed(SEED)
    per = -(-PAIRS // HOUSEHOLDS)
    rows = []
    for b in households("train", range(HOUSEHOLDS)):
        ev = [e for e in replay(b) if e[1] in e[2] and len(e[2]) > 1]  # hidden categories are not offered
        for q, gold, state, _ in rng.sample(ev, min(per, len(ev))):
            neg = rng.choice([c for c in state if c != gold])
            rows.append((q, state[gold], state[neg]))
    rng.shuffle(rows)
    print(f"{len(rows)} triplets from {HOUSEHOLDS} households; e.g. {rows[0]}", flush=True)
    model = SentenceTransformer(BASE, device="cuda" if torch.cuda.is_available() else "cpu")
    ds = Dataset.from_dict({"anchor": [r[0] for r in rows], "positive": [r[1] for r in rows], "negative": [r[2] for r in rows]})
    args = SentenceTransformerTrainingArguments(output_dir=str(OUT) + "_ckpt", num_train_epochs=EPOCHS, per_device_train_batch_size=BATCH,
                                                learning_rate=5e-5, warmup_steps=0.05, bf16=torch.cuda.is_available(), seed=SEED,
                                                logging_steps=200, save_strategy="no", report_to=[])
    SentenceTransformerTrainer(model=model, args=args, train_dataset=ds, loss=losses.MultipleNegativesRankingLoss(model)).train()
    model.save(str(OUT))
    print(f"-> {OUT}", flush=True)


def read():
    import torch
    from sentence_transformers import SentenceTransformer
    enc = os.environ.get("ENC", "")
    model = SentenceTransformer(str(ROOT / "models" / "encoders" / enc) if enc else BASE, device="cuda" if torch.cuda.is_available() else "cpu")
    if os.environ.get("READ", "households") == "budget":
        import real_budget_eval as RB
        budgets = [json.loads(RB.CACHE.read_text())["budget"]]
    else:
        a, z = map(int, os.environ.get("TEST_SEEDS", "100000-100049").split("-"))
        budgets = households("test", range(a, z + 1))
    cache, stats = {}, defaultdict(list)

    def emb(texts):
        new = [t for t in set(texts) if t not in cache]
        if new:
            for t, v in zip(new, model.encode(new, normalize_embeddings=True, batch_size=512, show_progress_bar=False)):
                cache[t] = v
        return np.stack([cache[t] for t in texts])
    for b in budgets:
        ev = replay(b)
        for q, gold, state, seen in ev:
            ids = list(state)
            if gold not in state:
                continue
            s = emb([state[c] for c in ids]) @ emb([q])[0]
            rank = int((s > s[ids.index(gold)]).sum())
            for g in ("all", "payee filed before" if seen else "first-time payee"):
                stats[g].append(rank)
    who = enc or BASE + " (untrained)"
    print(f"\n**{who}: right category by cosine (% top-1 / top-3 / top-10), {os.environ.get('READ', 'households')}**\n")
    print("| items | n | top-1 | top-3 | top-10 |"); print("|---|---|---|---|---|")
    for g, r in stats.items():
        r = np.array(r)
        print(f"| {g} | {len(r)} | {100 * (r < 1).mean():.1f} | {100 * (r < 3).mean():.1f} | {100 * (r < 10).mean():.1f} |")


if __name__ == "__main__":
    {"train": train, "read": read}[sys.argv[1]]()
