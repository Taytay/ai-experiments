# Dynamic Graph with Similarity-Aware Attention Graph Neural Network for Recommender Systems (DG-SA-GNN)

- arXiv 2605.05238, 8 May 2026 - https://arxiv.org/abs/2605.05238
- not listed on the page read
- no code or licence stated
- Source: the paper's arXiv / Hugging Face page read on 2026-10-07 (owner's request); full text not stored

## Summary
Builds four user-user similarity graphs (cosine, Jaccard, discounted Pearson, IP x IJ) and rebuilds them from the current embeddings at fixed epochs; fuses the four views with a graph transformer, refines users by cross-attention over their top-50 items, and samples 70% hard negatives from the top-200 scored items. MovieLens-100K only (943 users): Recall@20 0.1622 against LightGCN 0.1580, NDCG@20 0.0654 against 0.0663. Pairwise user similarity is quadratic.

## Relevance to this repo
Weak evidence (one small dataset, mixed result). The one transferable piece is hard negatives from the model's own top-scored wrong options, which our training does not do (every option of the day is a negative). Not a priority.
