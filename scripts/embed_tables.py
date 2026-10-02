"""Tables for PLAN step 157 (MODEL-25): embeddings judged against the generators' kinds (exp_embed_cluster.py). One row per reader and
read-out (pooling @ depth); payees by rendering (1-NN accuracy, k-means NMI), categories by the 1-NN among categories with another name
(meaning rather than name matching), by level, and k-means NMI / purity over all categories.
usage: uv run python scripts/embed_tables.py
"""
import glob
import json

NAMES = {"baseline": "no model (TF-IDF + SVD)", "decider-4b": "decider-4B, untrained", "Qwen3.5-4B-Base": "Qwen3.5-4B-Base",
         "Qwen3-Embedding-4B": "Qwen3-Embedding-4B (embedder)"}

if __name__ == "__main__":
    print("| reader | read-out | payees: name 1-NN | name + filings 1-NN | filings 1-NN | name + filings NMI | categories: 1-NN other name | "
          "0 rows | 1-2 rows | 3+ rows | NMI | purity |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for f in sorted(glob.glob("results/embed_cluster_*.json"), key=lambda f: ("baseline" not in f, "Embedding" in f, f)):
        d = json.load(open(f)); r = d["reader"]; name = NAMES.get(r, "decider-4B + recipe" if r.startswith("recipe") else r)
        pr = sorted({k.split(" | ")[1] for k in d["results"]})
        for ro in pr:
            p = d["results"].get(f"payees | {ro}", {}); c = d["results"].get(f"categories | {ro}", {})
            g = lambda blk, lv, m: blk.get(lv, {}).get(m, "–")  # noqa: E731
            print(f"| {name} | {ro} | {g(p, 'name', 'knn1')} | {g(p, 'name+filings', 'knn1')} | {g(p, 'filings', 'knn1')} | {g(p, 'name+filings', 'nmi')} | "
                  f"{g(c, 'all', 'knn1_othername')} | {g(c, '0 rows', 'knn1_othername')} | {g(c, '1-2 rows', 'knn1_othername')} | {g(c, '3+ rows', 'knn1_othername')} | "
                  f"{g(c, 'all', 'nmi')} | {g(c, 'all', 'purity')} |")
