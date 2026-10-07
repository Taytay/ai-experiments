# Graph Generative Pre-trained Transformer (G2PT)

- arXiv 2501.01073, Jan 2025 - https://arxiv.org/abs/2501.01073
- Chen, Wang, He, Du, Hassoun, Xu, Liu (Tufts)
- code https://github.com/tufts-ml/G2PT
- Source: the paper's arXiv / Hugging Face page read on 2026-10-07 (owner's request); full text not stored

## Summary
Generates graphs autoregressively: a graph is a token sequence (nodes as [type, id], then edges as [source, destination, type], ordered by a degree-based edge-removal process) and a decoder transformer (10M-300M parameters) is trained by next-token prediction, then tuned by rejection sampling or PPO toward goals. Molecules: MOSES validity 97.2% against DeFoG 92.8%, FCD 1.02 against 1.95; generic graphs competitive with diffusion models; property prediction ties GraphMAE (ROC-AUC 73.3). Order-sensitive, not permutation-invariant.

## Relevance to this repo
A graph generator (molecules, synthetic graphs), not a classifier or recommender; it does not apply to categorising transactions. If the task were cast as a graph, the relevant families are link prediction on the user-payee-category graph (ContextGNN, LightGCN, label propagation: references/graph_methods.md), not generation.
