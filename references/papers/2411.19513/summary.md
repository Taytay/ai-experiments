# ContextGNN: Beyond Two-Tower Recommendation Systems

- arXiv 2411.19513, 29 Nov 2024 - https://arxiv.org/abs/2411.19513
- Yuan, Zhang, He, Nitta, Hu et al. (Kumo.AI)
- CC BY 4.0; code https://github.com/kumo-ai/ContextGNN
- Source: the paper's arXiv / Hugging Face page read on 2026-10-07 (owner's request); full text not stored

## Summary
Two-tower recommenders score users and items from independent embeddings, so they miss pair-specific context (a repeat purchase against exploration); pair-wise models (NBFNet-style GNNs over the user's local subgraph) capture it but only for items near the user, and the paper's locality score (share of true items inside the user's k-hop subgraph) never exceeds 0.5 on its tasks. ContextGNN runs one GNN over the user-centric subgraph: items inside it are scored pair-wise (GNN readouts of user and item), items outside it by a two-tower score against shallow (learned ID) item embeddings, and an MLP on the user's representation predicts a per-user offset that fuses the two. Sampled softmax with ~1M negatives a batch. RelBench (8 tasks): MAP 9.23 against NBFNet 7.71 and the best two-tower 1.72; IJCAI temporal HR@1 0.411 against 0.148; Amazon-Book below UltraGCN (Recall@20 0.0451 against 0.0681), where locality is high. Gains shrink from 42-80% on low-locality tasks to 3-5% on high-locality ones. Shallow item embeddings make it transductive (no new items without a feature fallback).

## Relevance to this repo
The split it formalises is ours: a known payee is "local" (its own history decides, as fcr's history rows do, 97-98% on repeats), a first-time payee is "distant" (only knowledge of the name helps; decider and EmbeddingGemma 2 lead there, §207, §210). Its fusion (local score + learned per-user offset, tower score elsewhere) is a learned version of routing first-time payees to a second reader. Its shallow item embeddings do not transfer (our categories are per household, by text). Cheap test: fuse fcr and the EmbeddingGemma 2 two-tower per transaction, the weight set by whether the payee is known (locality), on the owner's budget.
