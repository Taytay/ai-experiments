"""Row 236 (a) (owner, 2026-10-07: "Should we just be running an embedding model at this point like the new Gemma? ... Either cosine
similarity or two towers approach"): untrained embedding models read the rational households' first purchases at a merchant by cosine
alone: the bank string (query prompt) against each of the household's category names (document prompt), the nearest wins. No history,
no training: what the base knows about merchant names and category names. Versions as rational_budgets.py (RATIONAL_PAYEES: "" real
held-out names, obvious invented, seenobv obvious names fcr trained on; RATIONAL_CATS). Synthetic data only.
env: ENCS (comma list of sentence-transformers models or models/encoders dirs), VERSIONS ("real obvious seenobv")
usage (EmbeddingGemma 2 needs sentence-transformers 6.1 / transformers 5.19 / torch 2.13):
  uv run --with "sentence-transformers>=6.1.0" --with "transformers>=5.18" --with torch==2.13.0 --with torchvision==0.28.0 python scripts/name_probe.py
"""
import importlib
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from ai_experiments.paths import ROOT  # noqa: E402

PROMPTS = {"google/embeddinggemma-2": ("task: classification | query: ", "title: none | text: ")}


def main():
    from sentence_transformers import SentenceTransformer
    encs = os.environ.get("ENCS", "google/embeddinggemma-2,BAAI/bge-small-en-v1.5,models/encoders/know_r212_e_k").split(",")
    versions = os.environ.get("VERSIONS", "real obvious seenobv").split()
    sets = {}
    for v in versions:
        os.environ["RATIONAL_PAYEES"] = "" if v == "real" else v
        import rational_budgets as RG
        importlib.reload(RG)
        sets[v] = RG.budgets("bank")
    print("| encoder | " + " | ".join(f"{v}: right first / top-3" for v in versions) + " |\n|---" * (len(versions) + 1) + "|")
    for enc in encs:
        path = str(ROOT / enc) if enc.startswith("models/") else enc
        m = SentenceTransformer(path, device="cuda")
        qp, dp = PROMPTS.get(enc, ("", ""))
        cells = []
        for v in versions:
            hit1, hit3, by_kind = [], [], defaultdict(list)
            for b in sets[v]:
                names = [c["name"] for c in b["categories"]]
                ids = [c["id"] for c in b["categories"]]
                D = m.encode([dp + n for n in names], normalize_embeddings=True)
                pn = {p["id"]: p["name"] for p in b["payees"]}
                first = [t for t in b["transactions"] if t["reason"][0] == "new"]
                Q = m.encode([qp + pn[t["payee_id"]] for t in first], normalize_embeddings=True, batch_size=256)
                S = Q @ D.T
                for t, s in zip(first, S):
                    order = [ids[j] for j in np.argsort(-s)]
                    hit1.append(order[0] == t["category_id"]); hit3.append(t["category_id"] in order[:3])
                    by_kind[t["kind"]].append(order[0] == t["category_id"])
            cells.append(f"{100 * np.mean(hit1):.1f} / {100 * np.mean(hit3):.1f}")
            worst = sorted(by_kind.items(), key=lambda x: np.mean(x[1]))[:4]
            print(f"  {enc} {v}: weakest kinds " + ", ".join(f"{k} {100 * np.mean(x):.0f}" for k, x in worst), flush=True)
        print(f"| {enc} | " + " | ".join(cells) + " |", flush=True)
        del m


if __name__ == "__main__":
    main()
