# PinnerSage: Multi-Modal User Embedding Framework for Recommendations at Pinterest

- arXiv 2007.03634 (v1, 7 Jul 2020) - https://arxiv.org/abs/2007.03634
- Aditya Pal, Chantat Eksombatchai, Yitong Zhou (equal contribution), Bo Zhao, Charles Rosenberg, Jure Leskovec (Pinterest Inc.)
- Venue: KDD 2020 (DOI 10.1145/3394486.3403280)
- Source: references/papers/2007.03634/paper_flat.tex

## One-paragraph summary
PinnerSage represents each user not by one vector but by a variable number of embeddings: 3-5 for light users, 75-100 for heavy ones. It runs Ward hierarchical clustering over the fixed PinSage embeddings of the user's last 90 days of repins and clicks, with a merge-distance threshold alpha. Each cluster is represented by its medoid, the member pin minimising the summed squared distance to the others. Each cluster gets a time-decayed importance, sum_i exp(-lambda (T_now - T_i)) with lambda = 0.01. Retrieval samples 3 medoids by importance and queries an HNSW index of pins. There is no learning: pin embeddings are fixed by design. A motivating oracle study shows single embeddings are poor: last pin is 0%, decay average +25%, k-means oracle with k=3 +98%, full-history oracle +140% (Table 1). In offline retrieval, Ward with medoids and lambda = 0.01 gives +110% relevance and +88% recall over last pin, vs +33%/+18% for its single-embedding variant and +31%/+16% for HierTCN (Table 3). A/B tests: Homefeed +4% engagement volume, Shopping +20%.

## Problem
A single user embedding averages unrelated interests into a point that represents none of them (Fig. 2: painting + shoes + sci-fi averages to "energy boosting breakfast"). Users switch between interests, and in Fig. 3 none of the last 5 pins resembles the latest. Earlier multi-embedding work fixed or capped the number of embeddings and never answered the production questions: how many embeddings, how to infer them at scale, how to choose which to use, and whether they help online.

## Method
- Design choices (Sec. 2):
  1. Pin embeddings are fixed (PinSage), not learned jointly with users, which would pull a user's unrelated interests together.
  2. No cap on the number of embeddings.
  3. Medoids rather than centroids: robust to outliers, no topic drift, stored as a pin id, and cacheable across users.
  4. Sample 3 medoids by importance for retrieval.
  5. Two-pronged updates: daily batch over 90 days plus lightweight online inference over the most recent 20 actions of the day, merged at day end.
  6. ANN retrieval.
- Clustering (Algorithm 1): Ward agglomerative clustering via Lance-Williams updates, with a nearest-neighbour-chain stack.
  - Distance update: d(Ci u Cj, Ck) = ((n_i+n_k) d_ik + (n_j+n_k) d_jk - n_k d_ij) / (n_i+n_j+n_k), Eq. 1.
  - Final clusters: merges with d_ij <= alpha, taken from a merge history sorted by decreasing distance, keeping disjoint ones.
  - Stated complexity O(m^2). The text defines m = |A|^2, which looks like a typo; the argument counts m initial clusters.
  - Appendix: proof that a cluster is never pushed twice onto the stack.
- Medoid (Eq. 2): argmin over m in C of sum_j ||P_m - P_j||^2.
- Importance (Eq. 3, Algorithm 2): sum over the cluster of exp(-lambda (T_now - T_i)). lambda = 0 means frequency only; lambda = 0.1 is recency-heavy; 0.01 was chosen. The time unit is not stated.
- Serving (Sec. 4):
  - HNSW over pins, with near-duplicate and low-quality pins filtered from the index.
  - Medoid-id caching.
  - Cost changes (Table 2): LSH Orthoplex to HNSW -60%; full index to refined index -50%; centroid to medoid -75%.
- Offline protocol:
  - Tens of millions of users, 90 days of actions and impressions, split by day d. The evaluation walks forward day by day and updates the models after each day.
  - Baselines: last pin; decay average (lambda 0, 0.01, 0.1, 0.25); LSTM, GRU and HierTCN trained to rank actions over impressions with several losses and negatives (impressions, random, popular, hard similar pins).
- Metrics:
  - Retrieval: floor(400/e) neighbours per embedding, at most 400 candidates. Relevance = share of future action pins with cosine >= 0.8 to any recommendation; recall = exact hits.
  - Ranking: actions vs 20 impressions per action, ranked by maximum cosine to any user embedding. Reported as R-Precision and reciprocal rank.

## Experiments and results
- Table 1 (next action, cosine >= 0.8 accuracy, lift over last pin): decay average +25%, k-means oracle (k=3) +98%, oracle (closest past pin) +140%.
- Table 3 (retrieval, lift over last pin, relevance / recall):

  | Model | Relevance | Recall |
  |---|---|---|
  | Decay average (lambda 0.01) | 28% | 14% |
  | HierTCN | 31% | 16% |
  | PinnerSage, sample 1 embedding | 33% | 18% |
  | K-means (k=5) | 91% | 68% |
  | Complete linkage | 88% | 65% |
  | Centroid | 105% | 81% |
  | HierTCN embedding per cluster | 110% | 88% |
  | importance lambda=0 | 97% | 72% |
  | importance lambda=0.1 | 94% | 69% |
  | Ward + medoid + lambda=0.01 | 110% | 88% |

- Table 4 (ranking, lift, R-Precision / reciprocal rank):
  - Decay average 8% / 7%; HierTCN 21% / 16%; sample-1 24% / 18%.
  - K-means 32% / 24%; complete linkage 29% / 22%.
  - Centroid, HierTCN and medoid all 37% / 28%.
  - lambda=0: 31% / 24%; lambda=0.1: 30% / 24%.
  - Ranking does not care how a cluster is embedded, but is sensitive to importance, which selects the 3 embeddings used.
- Fig. 7: relevance and diversity both rise with the number of embeddings used, e. Relevance gains taper past e=3, where recommendation diversity matches the diversity of the user's own actions.
- Table 5 (A/B vs decay-average single embedding): Homefeed volume +4%, propensity +2%; Shopping +20% / +8%.

## Limitations
- Offline results are lifts over a weak last-pin baseline, with no absolute numbers or error bars. Relevance (cosine >= 0.8 to any of up to 400 candidates) is a lenient proxy.
- No learning at all: quality is bounded by PinSage, and there is no task-specific training of the clustering or importance. The single-cluster sequence-model variant ties medoids, so learning adds nothing as used.
- The choice of the threshold alpha and its sensitivity are not reported. The lambda unit is unstated. Medoid sampling makes recommendations random.
- Two years later PinnerFormer (2205.04507, Table 1) found a single learned embedding beats a 20-cluster PinnerSage oracle 5x on 14-day Recall@10 (0.229 vs 0.046), though PinnerSage stays more diverse.

## Relevance to this workspace
- **Our MaxSim encoder is PinnerSage at the category level.** It represents a category by the set of its filings and scores the mean of the 3 best cosines (69.6%). The paper's argument applies directly: categories are purposes, and one purpose ("Kids", "House", a trip) holds several merchant modes (school fees, toys, paediatrician). A centroid of those lands nowhere, which is the "energy boosting breakfast" failure, and matches our prototype reader's 48%. Two cheap, training-free variants to test against MaxSim:
  - Ward-cluster each category's filings (bge-small vectors, threshold alpha tuned on synthetic data), score max cosine to medoids, and weight by cluster importance with exponential time decay. The importance term adds a recency and frequency prior inside each category, which plain MaxSim lacks.
  - Report centroid vs medoid vs top-3 MaxSim side by side. The paper found centroid only slightly worse than medoid (105% vs 110%) once clusters exist, so the gain comes from clustering, not from medoids.
- **The decay weighting has a direct analogue.** Pure frequency (lambda=0) and heavy recency (0.1) both lost to 0.01 by 13-16 points of relevance. For our category prior and kNN vote, weight filings by exp(-lambda x age) and sweep lambda per day on synthetic data. Long-lived purposes (rent) need frequency; trips need recency.
- **Choosing the decider's history rows.** The decider sees 24 history rows. PinnerSage's diversity result (Fig. 7: diversity matched the user's own at e=3) suggests filling some of those slots with cluster medoids of the user's whole history: one representative row per interest cluster, sampled by importance. This gives breadth, while the nearest-neighbour rows give depth. This is a prompt-construction change with no training.
- **Fixed item embeddings.** Their first design choice (no joint training, so unrelated interests are not pulled together) is a caution for the planned joint graph or category embeddings. When a household's unrelated categories are trained together, their embeddings may merge. Keep a fixed-encoder kNN/MaxSim path as the robust baseline.
- **Does not transfer:** ANN infrastructure at billions of pins, cost tables, and the two-pronged batch/online update (our per-user data is small enough to recluster on every new filing). The next-action "cosine >= 0.8" metric does not apply to a per-user softmax over categories.

## Key references worth following up
- Ward 1963 (minimum-variance hierarchical clustering); Lance and Williams 1967.
- Weston et al. 2013, nonlinear latent factorisation (multiple user vectors, +25% on YouTube).
- You et al. 2019, HierTCN (hierarchical temporal convolution for user sequences).
- Ying et al. 2018, PinSage.
