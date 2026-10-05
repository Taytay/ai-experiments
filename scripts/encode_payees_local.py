"""PLAN step 186: embed a real budget's strings with a trained payee encoder (train_payee_encoder.py) on this machine's CPU, so the strings
never leave it. Writes, privately (0600), next to real_budget_eval.py's files:
  raw_emb_<ENC>.npz    every raw bank string YNAB imported (payee_resolution_real.py EMB_FILE=raw_emb_<ENC>.npz)
  payee_emb_<ENC>.npz  the budget's payee names as the items show them (real_budget_eval.py's similar-payee retrieval, EMB_FILE)
env: BUDGET, CACHE_PATH / OUT (as real_budget_eval.py), ENC (payee_enc_v1: models/encoders/<ENC>).
usage: BUDGET=<id> uv run python scripts/encode_payees_local.py
"""
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import real_budget_eval as RB  # noqa: E402

from ai_experiments.paths import ROOT  # noqa: E402

ENC = os.environ.get("ENC", "payee_enc_v1")

if __name__ == "__main__":
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(str(ROOT / "models" / "encoders" / ENC), device="cpu")
    b = json.loads(RB.CACHE.read_text())["budget"]
    raw = sorted({t["import_payee_name_original"] for t in b["transactions"] if not t.get("deleted") and t.get("import_payee_name_original")})
    items = json.loads((RB.OUT / "items.json").read_text())["items"]
    names = sorted({it["prompt"].rsplit("Transaction: ", 1)[1].split(" | ")[1] for it in items})
    for tag, xs in (("raw_emb", raw), ("payee_emb", names)):
        vecs = model.encode(xs, normalize_embeddings=True, batch_size=256, show_progress_bar=False)
        p = RB.OUT / f"{tag}_{ENC}.npz"
        np.savez(p, names=np.array(xs), vecs=vecs.astype(np.float32)); os.chmod(p, 0o600)
        print(f"{len(xs)} strings -> {p.name}", flush=True)
