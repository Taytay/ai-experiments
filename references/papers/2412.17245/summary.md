# GraphHash: Graph Clustering Enables Parameter Efficiency in Recommender Systems

- arXiv 2412.17245, 2024 (arXiv Dec 2024) - https://arxiv.org/abs/2412.17245
- Wu, Loveland, Chen, Liu, Chen, Neves, Jadbabaie, Ju, Shah, Zhao (Snap, MIT)
- code https://github.com/snap-research/GraphHash; licence not stated in the abstract page
- Source: the paper's arXiv / Hugging Face page read on 2026-10-07 (owner's request); full text not stored

## Summary
Embedding tables for user and item IDs dominate recommender memory; the hashing trick shares rows between IDs at the cost of collisions. GraphHash assigns buckets by Louvain modularity clustering of the user-item bipartite graph, so entities that share a row also share interaction patterns; the paper argues modularity maximisation is a cheap proxy for message-passing smoothing. DoubleGraphHash adds a random hash to cut collisions (for CTR). Retrieval (Gowalla, Yelp2018, Amazon-Book; MF, NeuMF, LightGCN): +101.5% Recall@20 and +88.3% NDCG@20 over double frequency hashing at 75%+ fewer embedding parameters; CTR: 2.9% better log-loss, +0.2% AUC. Louvain takes ~2 s on Gowalla.

## Relevance to this repo
We have no ID embedding tables (payees and categories are read as text), so the compression has no target here. The idea that carries over is the clustering: payees that many users file alike, clustered on the payee x (user's category kind) graph, give an unknown payee's string a behaviour cluster from other users (row 189's crowd behaviour vectors were a first form, with 4.6-5.1% noisy crowd lines). That needs many real users' filings: worth revisiting when the owner's real data arrives.
