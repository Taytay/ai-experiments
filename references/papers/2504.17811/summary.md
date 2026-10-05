# OmniSage: Large Scale, Multi-Entity Heterogeneous Graph Representation Learning

- arXiv 2504.17811 (v3, 12 Jun 2025; v1 22 Apr 2025) - https://arxiv.org/abs/2504.17811
- Anirudhan Badrinath, Alex Yang, Kousik Rajesh, Prabhat Agarwal, Jaewon Yang, Haoyu Chen, Jiajing Xu, Charles Rosenberg (Pinterest)
- Venue: KDD 2025 Industry Track (Proceedings of the 31st ACM SIGKDD, V.2, DOI 10.1145/3711896.3737253)
- Source: references/papers/2504.17811/paper_flat.tex

## One-paragraph summary
OmniSage is the successor to PinSage at Pinterest. It learns one embedding per entity type (Pin, board) that combines three signals: the graph (a heterogeneous Pin-board / Pin-Pin graph), content (image and text features), and user activity sequences. Each node is embedded by a small transformer. Its input is the node's own features plus the features of neighbours chosen by random walk with restart (RWR), ordered by visit probability and taken top-k per node type and per edge-type subset; a CLS token gives the output. Three sampled-softmax contrastive tasks train it jointly: entity-entity (graph and engagement pairs), entity-feature (the embedding must retrieve its own and its neighbours' raw features), and user-entity (a causal transformer over the user's sequence of OmniSage embeddings predicts the next and future items, PinnerFormer-style, with gradients flowing into the embedders). The graph has 5.6B nodes and 63.5B edges after degree pruning. Against PinSage, entity-entity recall@10 rises 0.461 to 0.609 (+32%); against PinnerFormer, user-entity recall@10 rises 0.306 to 0.580 (+89.5%) (Table 2). Removing the graph drops these to 0.440 and 0.375 (Table 3). Online, the reported A/B lifts sum to about 2.5% in sitewide repins across five applications.

## Problem
Graph, sequence and content representation learners are usually built and served separately, and each use case gets its own model. The paper wants one universal entity representation that serves many applications. That requires scaling GNN neighbour sampling and feature fetching to billions of nodes, handling node and edge heterogeneity, and combining graphs with sequences, which need different architectures.

## Method
- Heterogeneous graph H = (V, E, T^v, T^e) with typed nodes and typed edges (Section 3.1). Production edges (Section 4.1):
  - Pin-board: the Pin is saved in the board.
  - Pin-Pin: a query Pin on the related-Pins surface led to a save or click on the other Pin.
  - Undirected, built from engagement logs. Users are not nodes.
- Graph pruning (Appendix, Algorithm 2): node u of degree d_u keeps d_target = max(min(d_u^alpha, d_max), d_min) uniformly sampled edges. alpha = 0.86, d_min = 10, d_max = 10000. Low-degree nodes are untouched; the target is hub redundancy.
  - Graph stats (Table 1): 3.5B Pins (average degree 21.7), 2.1B boards (24.3), 51.4B Pin-board and 12.1B Pin-Pin edges.
- Neighbour sampling (Section 3.1.1): an importance neighbourhood from PinSage, extended to heterogeneous graphs.
  - For each subset of relation types R_i, take the induced subgraph H_R and compute RWR proximity s(u, v, H_R) with forward-push (Andersen et al. 2006), error threshold delta = 1/B.
  - Keep the top k_t nodes of each node type t, then take the union over subsets.
  - Production: top 25 Pins + 75 boards from the {Pin-Board} subgraph and top 50 Pins from {Pin-Pin}. Separate quotas keep the rarer Pin-Pin edges from being drowned out.
- Embedder (Section 3.2, Appendix A.3):
  - One embedder f_t per entity type.
  - Each node has image features (in-house ViT) and text features (shared hash embedding over n-gram tokenised text, Svenstrup et al. 2017).
  - Neighbours are ordered by proximity score, each passed through an MLP, and concatenated into a sequence with the anchor node's features and a CLS token.
  - The encoder is 1 transformer layer: 12 heads, d = 768, MLP 3072, no causal mask. The CLS output goes through an MLP head and L2 normalisation.
  - Pins and boards share the architecture; boards have a different sequence length.
- Generic loss (Eq. 1): sampled softmax with in-batch negatives plus random negatives.
  - Score s(q, v) = lambda * q^T v - log Q(v), a logQ sampling-bias correction.
  - Q(v) is estimated with a count-min sketch.
- Entity-entity (Eq. 2): (q, p) pairs come from graph-neighbourhood samples and from engagement logs (Pin-Pin, board-Pin). Logs are segmented (promoted, shoppable, video) and weighted per source w_i. Negatives are random nodes of the positive's type.
- Entity-feature (Eq. 3): for each feature type x, a 4-layer ReLU MLP g_x (hidden 1024, no normalisation) encodes the raw feature of the node and of its sampled neighbours. That encoding is the query; the node's embedding f_t(u) is the positive; other nodes are negatives. The purpose is to keep content in the embedding.
- User-entity (Eqs. 4-5):
  - A causal transformer T (4 layers, 4 heads, d = 512, GELU) runs over the user's sequence of OmniSage embeddings P_tau. Output U_tau is the user embedding at step tau.
  - Next-action loss: U_t retrieves P_{t+1}.
  - Future-action loss: every U_t retrieves an item sampled uniformly from the window (t_max, t_max + Delta).
  - Both use in-batch negatives, and gradients flow back into the embedders.
- Overall loss (Eq. 6): lambda_p L_pair + lambda_f L_feat + lambda_s (L_next + L_fut).
- Training infrastructure:
  - "Grogu" graph engine: an in-memory, memory-mapped C++ adjacency list for neighbour sampling, plus a RocksDB key-value featurizer. Features are kept even for pruned nodes.
  - Chunked sampled softmax (Algorithm 1): detach Q, P, N; compute the logits and cross-entropy chunk by chunk; accumulate gradients with respect to the embeddings; then backpropagate once through the model. Memory no longer grows with the number of negatives, and no extra backward passes are needed (GradCache-like).
  - Negatives are aggregated across devices.
  - AdamW, lr 0.005, linear warmup then cosine, fp16.
- Serving: daily offline batch inference over 6B+ entities. Pruned or isolated nodes get zero neighbour features.

## Experiments and results
- Metric: recall@10 of the positive among a large set of random negatives. Evaluation sets are disjoint by user and/or pair; the sequence task uses future Pins from 28 days after the training window.
- Main comparison (Table 2):
  - OmniSage: Entity-Entity 0.609, Entity-Feature 0.338, User-Entity 0.580.
  - PinSage: 0.461 (Entity-Entity only).
  - PinnerFormer: 0.306 (User-Entity only).
- Graph construction (Table 3, Entity-Entity / User-Entity):
  - No graph: 0.440 / 0.375.
  - Pin-Board only: 0.529 / 0.564.
  - Pin-{Board, Pin}: 0.609 / 0.580.
- RWR budget (Table 4): B/2 0.609 / 0.576; B 0.609 / 0.580; 2B 0.612 / 0.576. Results are robust to the budget.
- Task combinations (Table 5):
  - Entity-Entity alone: 0.623.
  - Entity-Feature alone: 0.373.
  - User-Entity alone: 0.564.
  - Entity-{Entity, Feature}: 0.628 / 0.379.
  - All three: 0.609 / 0.338 / 0.580.
  - The sequence task gains from the other tasks; the pair and feature tasks lose a little.
- Downstream Homefeed ranking (Table 6), hits@3 lift over PinSage, repin / longclick:
  - Entity-Entity: neutral / neutral.
  - Entity-{Entity, Feature}: +0.5% / +0.7%.
  - All three tasks: +2.6% / +1.5%. Above 0.5% counts as significant.
- Online Pin embeddings (Table 7), sitewide repins: HF ranking +0.92%, related-Pins retrieval +0.39%, related-Pins ranking +0.85%. Surface repins +1.22 / +0.47 / +1.35%.
- Online board embeddings on Board More Ideas (Table 8): ranking +0.67% sitewide, +5.91% BMI repins; retrieval +0.38% / +3.79%.
- Appendix ablations:
  - Without the graph (Table 9): Entity-Entity alone 0.471; User-Entity alone 0.330; joint 0.440 / 0.375.
  - Pruning alpha (Table 10): 0.5 gives 13.6B edges, 0.584 / 0.584; 0.7 gives 26.4B, 0.592 / 0.582; 0.86 gives 64.5B, 0.609 / 0.576. A bigger graph helps pair retrieval and slightly hurts sequence retrieval. (The main text gives 63.5B edges and 0.580 for alpha = 0.86.)
  - GAT aggregation instead of the transformer (Table 11): -6.0% (1-hop), -8.5% (2-hop) on entity-entity recall.
  - Random negatives (Table 12): +31.2% user-entity recall with 1K random negatives and +43.5% with 10K, against none. In-batch negatives without random negatives drop recall on random-corpus retrieval.
  - Halving the RWR top-k neighbours (Table 13): Entity-Entity -0.1%, User-Entity -1.4%.

## Limitations
- Only Pins and boards are deployed. Users, queries and items are motivated in the text but not evaluated as node types.
- Offline metrics are recall against random negatives, an easy regime. There is no hard-negative or fine-grained evaluation.
- Many choices (edge-type subsets R_i, per-type k_t, task weights, segment weights) are set "based on empirical analysis and domain knowledge" and not reported.
- There are no absolute online numbers, no variance across seeds, and no comparison with LightGCN or other non-RWR graph embeddings.
- Cold-start nodes get zero neighbour features. Graph augmentation for them is left to future work.
- Pruning results are slightly inconsistent between tables (edge count and user-entity recall at alpha = 0.86).

## Relevance to this workspace
- This is a worked template for the planned PinSage/GraphSAGE row (PLAN row 198). Mapping:
  - Pin becomes a bank string (payee).
  - Board becomes a cross-household category cluster (our 64 k-means clusters) or a (household, category) node.
  - The Pin-board edge becomes "some household filed this payee into this cluster".
  - The Pin-Pin edge becomes two payees co-filed into the same category by the same household, or consecutive filings.
  - Their recipe for a node embedding is: RWR top-k neighbours per type, ordered by visit probability, fed with the node's own text features to a 1-layer transformer with a CLS token. That is a direct replacement for GraphSAGE mean pooling, and GAT was 6-8.5% worse (Table 11).
  - The graph is worth +0.17 entity-entity and +0.2 user-entity recall over no graph (Table 3). That is a reason to expect the graph to help first-time payees, which is where our crowd line already gives +7 to +12.
- Per-type quotas (25 Pins + 75 boards, plus 50 Pins from a separate edge set) matter because the rare edge type gets drowned otherwise. For us, quota payee neighbours and cluster neighbours separately, so frequent clusters do not crowd out co-filed payees.
- Degree pruning d^alpha (alpha 0.86, d_min 10) is the right tool for hub payees ("AMAZON", "PAYPAL", "VENMO"): such a node links to every cluster and turns RWR into noise. Prune or downweight edges at hubs before sampling.
- Negatives are the most actionable point:
  - Random negatives gave +31% (1K) and +43.5% (10K), and in-batch-only training hurt retrieval over a random corpus (Table 12).
  - Our encoders use in-batch (other households) plus one hard negative. Try adding a pool of 1K to 10K random negatives (random transactions or category texts from other households) with a logQ correction, since in-batch positives over-sample popular payees and categories (s = lambda q^T v - log Q(v), with Q from counts).
  - Chunked softmax (Algorithm 1) makes thousands of negatives fit on the 3090 for bge-small.
  - Caveat: our test asks for the right category among one household's 20 to 100 categories, a hard-negative regime. Random negatives help corpus-wide calibration, not fine discrimination, so keep the per-household hard negatives too.
- The entity-feature task (embedding must retrieve its own raw features) is a cheap regulariser that keeps text content in a graph embedding. It matters for new bank strings that have no edges yet. If we train a graph embedding for payees, add it so an unseen payee's text-only embedding stays in the same space.
- The user-entity task is a PinnerFormer-style household sequence model: next filing and a future window. It is the lightweight version of the per-user state vector (row 196; see also PinFM). OmniSage shows it does better when trained jointly with the pair tasks (0.564 to 0.580) and that it adds most downstream (Table 6: +2.6% vs +0.5%).
- Hash embeddings over n-grams for text are a cheap, robust way to encode noisy bank strings. They are worth trying as an extra feature beside bge, because character n-grams survive "SQ *" and "TST*" prefixes and store numbers.
- Does not transfer:
  - The infrastructure: Grogu, RocksDB, billions of nodes, daily batch inference over 6B entities. Our graph fits in memory, and networkx or PyG is enough.
  - Per-household category names are not shared nodes, so only clusters and payees can be graph nodes. Personal categories still need the history layer (kNN, MaxSim, decider) on top.
  - Recall@10 against random negatives would overstate gains for us. Judge on blind_v1/v2 and the owner's budget as usual.
