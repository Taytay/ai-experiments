"""Linear probes on the categoriser's hidden states (PLAN step 67, MODEL-8): where the decision forms and what the weights store, before
and after database episodes. Qwen3.5-2B (Apache-2.0; row 78) untrained, trained without a database, and trained with database episodes
(adapters by name under models/adapters); every REAL-6 item's prompt (no record) is run once and the hidden state of every layer is
taken at two positions:

  stmt  the last token of the query's statement string (before " | $"), what the model has read about the merchant
  cue   the final "Category:" token, where the answer is read out

Probes (logistic regression, standardised features), per layer and position:

  std    the merchant's standard category (REAL-6's 12), 5-fold cross-validated with merchants grouped, on the merchants any user
         labelled; then trained on all of them and read on the DB-only merchants (never in a history; their category reaches the weights
         only through database episodes)
  gold   the standard category of the user's gold label at the cue (does the cue carry the answer's meaning?), grouped folds by user

Writes results/probes_<model or adapter>.json. Tracker experiment "probes".
usage: WHAT=base|<adapter> MODEL=Qwen/Qwen3.5-2B uv run [Qwen3.5 overlay] python scripts/exp_probes.py
"""
import json
import os
import time

import numpy as np

from ai_experiments import real6 as R6
from ai_experiments import transactions as T
from ai_experiments.evals.tracker import Run
from ai_experiments.paths import ROOT

WHAT, MODEL = os.environ.get("WHAT", "base"), os.environ.get("MODEL", "Qwen/Qwen3.5-2B")
BATCH = int(os.environ.get("BATCH", "8"))
OUT = ROOT / "results" / f"probes_{MODEL.split('/')[-1] if WHAT == 'base' else WHAT}.json"


def features(items):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    base = MODEL if WHAT == "base" else json.loads((ROOT / "models" / "adapters" / WHAT / "adapter_config.json").read_text())["base_model_name_or_path"]
    tok = AutoTokenizer.from_pretrained(base)
    lm = AutoModelForCausalLM.from_pretrained(base, dtype=torch.bfloat16).cuda().eval()
    if WHAT != "base":
        from peft import PeftModel
        lm = PeftModel.from_pretrained(lm, str(ROOT / "models" / "adapters" / WHAT)).merge_and_unload().eval()
    inner = lm.model
    feats = {"stmt": [], "cue": []}
    with torch.no_grad():
        for k in range(0, len(items), BATCH):
            chunk = items[k:k + BATCH]; rows, pos = [], []
            for it in chunk:
                p = it["prompt"]; q = p.rindex("Transaction: ")
                stmt_end = p.index(" | $", q)
                ids = tok(p, add_special_tokens=False, return_offsets_mapping=True)
                off = ids["offset_mapping"]
                s = max(t for t, (a, b) in enumerate(off) if b <= stmt_end)
                rows.append(ids["input_ids"]); pos.append((s, len(ids["input_ids"]) - 1))
            L = -(-max(map(len, rows)) // 64) * 64  # few shapes for fla's per-shape kernels
            x = torch.tensor([r + [tok.pad_token_id or 0] * (L - len(r)) for r in rows], device="cuda")
            att = (torch.arange(L, device="cuda")[None] < torch.tensor([len(r) for r in rows], device="cuda")[:, None]).long()
            hs = inner(input_ids=x, attention_mask=att, output_hidden_states=True).hidden_states  # embeddings + each layer
            for j, (s, c) in enumerate(pos):
                feats["stmt"].append(torch.stack([h[j, s] for h in hs]).float().cpu().numpy())
                feats["cue"].append(torch.stack([h[j, c] for h in hs]).float().cpu().numpy())
    return {k: np.stack(v) for k, v in feats.items()}  # [items, layers + 1, hidden]


def probe(X, y, groups, train_mask=None, test_mask=None):
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    clf = lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=300, C=0.1))  # noqa: E731
    if train_mask is not None:
        m = clf().fit(X[train_mask], y[train_mask])
        return float((m.predict(X[test_mask]) == y[test_mask]).mean())
    acc = []
    for tr, te in GroupKFold(5).split(X, y, groups):
        m = clf().fit(X[tr], y[tr]); acc.append((m.predict(X[te]) == y[te]).mean())
    return float(np.mean(acc))


if __name__ == "__main__":
    from ai_experiments.licences import open_licence
    open_licence(MODEL)
    doc = R6.load(); items = doc["items"]
    std = {m["name"]: m["category"] for m in T.load()["merchants"]}
    users = {u["user"]: u for u in doc["users"]}
    db_only = R6.db_only_merchants()
    t0 = time.time(); F = features(items); t_feat = time.time() - t0
    y_std = np.array([std[it["merchant"]] for it in items])
    gold_std = []
    for it in items:
        cat = next(c for c in users[it["user"]]["categories"] if c["name"] == it["options"][it["answer"]].strip())
        gold_std.append(cat["standard"][0] if len(cat["standard"]) == 1 else "multi")
    y_gold = np.array(gold_std)
    merch = np.array([it["merchant"] for it in items]); user = np.array([it["user"] for it in items])
    dbo = np.array([m in db_only for m in merch]); lab = ~dbo
    res = dict(model=MODEL, what=WHAT, n_items=len(items), n_db_only=int(dbo.sum()), layers=int(F["stmt"].shape[1]), feature_minutes=round(t_feat / 60, 1), probes={})
    for pos in ("stmt", "cue"):
        for layer in range(F[pos].shape[1]):
            X = F[pos][:, layer]
            r = dict(std_cv=probe(X[lab], y_std[lab], merch[lab]), std_db_only=probe(X, y_std, merch, train_mask=lab, test_mask=dbo))
            if pos == "cue":
                r["gold_cv"] = probe(X, y_gold, user)
            res["probes"][f"{pos}_{layer}"] = r
            print(pos, layer, r, flush=True)
    with Run("probes", model=MODEL, config=dict(what=WHAT, model=MODEL)) as run:
        run.log({f"{k}_{m}": v for k, d in res["probes"].items() for m, v in d.items()})
    OUT.write_text(json.dumps(res, indent=2)); print("->", OUT)
