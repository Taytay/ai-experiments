# Synergizing Implicit and Explicit User Interests: A Multi-Embedding Retrieval Framework at Pinterest

- arXiv 2506.23060 (v1, 29 Jun 2025) - https://arxiv.org/abs/2506.23060
- Zhibo Fan, Hongtao Lin, Haoyu Chen, Bowen Deng, Hedi Xia, Yuke Yan, James Li (Pinterest)
- Venue: KDD '25 (DOI 10.1145/3711896.3737265)
- Source: references/papers/2506.23060/paper_flat.tex

## One-paragraph summary
A single two-tower user embedding is pulled toward the user's dominant interest and misses torso and tail interests. The paper builds several conditional user embeddings, f(i|u,c) proportional to exp(phi(u,c)^T psi(i)), from two sources:
- Implicit conditions: K_im=7 clusters of the user's engagement sequence, found by a Differentiable Clustering Module (DCM). DCM is MIND-style capsule routing with validity-aware farthest-point initialisation and single-assignment routing. Each training positive is associated with the argmax cluster embedding.
- Explicit conditions: K_ex=5 followed topics, via Conditional Retrieval (2508.16793). The condition is logged with the engagement ("source interest") instead of sampled from the item's topics.

Candidates from each embedding are merged round-robin. DCM beats self-attention, interest-token, MIND and PinnerFormer-subsequence variants offline (HR@100 0.185, Table 2) and online (+0.86% Homefeed repins, Table 5). Source-interest CR beats item-interest CR (filtered HR@100 0.191 vs 0.164, Table 3). The two halves retrieve only 3.2% overlapping candidates and help different users: explicit helps non-core users, implicit helps core users. Together: sitewide repins +0.48%, Homefeed repins +1.09%, adopted Pincepts +0.81% (Table 6).

## Problem
Retrieval sets the recall ceiling, and Homefeed has no query, so coverage of a user's many interests is the goal. Two-tower models have no user-item interaction before the dot product, and the head interest dominates the single embedding (Fig. 1). The authors frame both fixes as conditional representation learning with two design questions. Condition construction: how interests are formed and encoded. Condition association: which condition each positive item is trained against.

## Method
- Framework (Section 3.1, Fig. 2): K_im implicit plus K_ex explicit conditions, each producing one user embedding phi(u,c). Items use one psi(i).
- Capsule background (Section 3.2.1, Eqs. 1-3): MIND-style dynamic routing with b_ij = softmax_j(c_j^T S e_i), c_j = squash(sum_i b_ij S e_i), iterated.
- DCM construction (Section 3.2.2, Fig. 3):
  - item vectors e_i come from a 2-layer GELU MLP over concatenated item features (PinSage and categorical features; Eq. 4), not a shared bilinear S;
  - Validity-Aware Farthest Point Initialisation: the first centroid is random, then each next centroid i* = argmax_i min_j -I_valid(e_i) c_j^T e_i (Eqs. 5-6). Invalid items (missing features, negative actions) are masked because their embeddings are out of distribution and would otherwise be picked as centroids;
  - Single-Assignment Routing (from MIND360): b_ij is the softmax weight kept only for the argmax centroid and zeroed elsewhere (Eq. 7), which pushes centroids apart.
- Implicit association (Section 3.2.3): j* = argmax_j o_u^j^T o_{y_i} (Eq. 8). Sampled-softmax loss with logQ correction from streaming frequency estimation, using only o_u^{j*} against in-batch negatives (Eq. 9). This is cheaper than UMI, which scores negatives against all embeddings.
- Alternatives (Section 3.2.4):
  - self-attention heads (ComiRec);
  - interest tokens (MVKE without gating, i.e. a single-head decoder with learnable query tokens);
  - PFS: K-means of 10M pins into 32 PinSage clusters, the user sequence split into per-cluster subsequences, PinnerFormer applied to each, used as conditions in a two-tower model (better than retrieving with them directly; Appendix B).
  - With argmax association, self-attention and interest-token conditions collapse ("winner-takes-all": one embedding gets gradients and shared parameters make them identical). A Straight-Through Gumbel-Softmax is needed for them to converge. DCM avoids collapse because its conditions come from clustering the sequence.
- Explicit CR (Section 3.3, Fig. 4): the condition (topic) embedding goes into the user tower's embedding layer and then through feature crossing. Association happens at logging time: engagements served by the old followed-topic inverted-index retriever are logged as (condition, user, item). This replaced the original CR's random sampling from item topics, which can mismatch user intent. A post-filter on item-to-topic acts as a relevance guardrail.
- Feature crossing (Appendix A): DHEN with two levels. Level 1: a 2-layer Transformer (256 dims, 4 heads) in parallel with a 2-layer MLP (1024). Level 2: a 4-block parallel MaskNet (128, ratio 0.5) in parallel with an MLP (1024). Outputs are summed per level, and features are split into equal-dimension fields for the transformer.
- Deployment (Section 3.5): ANN per embedding. The implicit budget is proportional to the cluster weight sum_i b_ij. The explicit side randomly samples K_ex followed topics with equal budget. Round-robin merge with dedup, total budget on par with single-embedding. p90 latency 150 to 205 ms.
- Data: a 15-day log window (14 train, 1 eval), 6B engagements, 160M users. Composite positives (click, repin, ...). Eval corpus is the 1M most-engaged items. An item's score under multiple embeddings is the maximum over embeddings. HR@K.

## Experiments and results
- Table 2, implicit offline (HR@100 / HR@1000): self-attention 0.167 / 0.470; interest token 0.180 / 0.474; MIND 0.175 / 0.464; DCM 0.185 / 0.476.
- Table 3, explicit offline (filtered HR@100 / filtered HR@1000 / unfiltered HR@100): CR with item interest 0.164 / 0.541 / 0.139; CR with source interest 0.191 / 0.565 / 0.145.
- Table 4, explicit online, columns as best read (the header layout is ambiguous):
  - CR with filter vs inverted index: Homefeed repins +0.56%, non-core repins +1.13%, adopted Pincepts +0.42%, non-core Pincepts +0.44%
  - CR without filter vs inverted index: +0.3% (not significant), +0.46% (not significant), +0.37%, +0.42% (not significant)
  - source vs item interest: repins +0.98%, non-core repins +3.04%, Pincepts +0.32% (not significant), non-core Pincepts +1.03%
- Table 5, implicit online, on top of shipped CR (repins all / core; Pincepts all / core):
  - self-attention +0.83 / +1.01; +0.44 / +0.63
  - interest token +0.68 / +0.95; +0.21 / +0.19 (not significant)
  - MIND +0.43 (not significant) / +0.56; +0.04 / +0.15 (not significant)
  - PFS +0.47 / +1.02; +0.32 / +0.46
  - DCM +0.86 / +1.23; +0.46 / +0.87
- Table 6, whole framework: sitewide repins +0.48%, Homefeed repins +1.09%, adopted Pincepts +0.81%.
- Table 7, DCM ablation (VA-FPI, SAR, number of clusters; HR@100, repins, Pincepts):
  - neither, 7: 0.175, +0.43 (n.s.), +0.04 (n.s.)
  - SAR only: 0.176, +0.37 (n.s.), +0.30 (n.s.)
  - VA-FPI only: 0.175, -0.10 (n.s.), +0.17 (n.s.)
  - both, 2 clusters: 0.176, -0.53% (significant loss), +0.25 (n.s.)
  - both, 4 clusters: 0.180, -0.15 (n.s.), +0.29 (n.s.)
  - both, 7 clusters: 0.185, +0.86, +0.46
  - Both modifications are needed, and too few clusters hurts.
- Per-user adaptive cluster count: -0.12% repins vs fixed. Taking the global top candidates across embeddings instead of round-robin: -0.35% repin users.
- Fig. 5 (t-SNE): without validity filtering some centroids collapse to one point.
- Overlap between explicit and implicit candidates: 3.2%.
- Appendix C, Table 8, explicit CR by non-core cohort (repins / Pincepts): casual +1.12 / +0.23; marginal +1.46 / +0.05; resurrected +1.00 / +0.67; new +0.56 / +0.65 (new users are under 10% of non-core, not significant).
- Case study (Fig. 6): CR recovers "education", which DCM missed. DCM learns personalised granularity (coarse "beauty", fine "candid youthful moments" photography).

## Limitations
- Offline gains are small (DCM vs MIND: +0.010 HR@100). Decisions were made mostly online over 2 weeks, so offline results alone would not have chosen DCM.
- K_im=7 and K_ex=5 are fixed globally. The adaptive attempt did not help, but only one variant was tried.
- Label sampling and weighting are omitted "for brevity". The eval corpus (top 1M items) is biased toward the head.
- Explicit-condition association needs a pre-existing retriever that logs source topics, which is a bootstrapping dependency.
- No numbers for the collapse claim (Gumbel fix) beyond the text. Table 4's layout is ambiguous.

## Relevance to this workspace
- The construction/association framing fits our problem cleanly. Our explicit conditions are the user's categories, and association is free and exact: every filing records its category, which is the "source interest" the paper found better than conditions inferred from the item (Table 3: +0.027 filtered HR@100; Table 4: +3.04% for non-core users). The planned row 196 (condition tokens per category, user tower) should therefore train each transaction against its own filed category, never against an inferred cluster.
- Multi-embedding with max-over-embeddings scoring is what our MaxSim already does: one embedding per category with the max over its filings, close to "item score = max over user embeddings". The paper's improvement is to learn the per-condition embedding through user-condition feature crossing (CR) rather than raw filing vectors. A per-category embedding phi(u, c) = UserTower(user state, category text or prototype), trained with InfoNCE against the transaction, is the direct port.
- Implicit DCM suits payee or behaviour clusters inside one category, e.g. a "Groceries" category that is really two shops plus warehouse-club runs. Run farthest-point initialisation and single-assignment routing on a category's filings to get several prototypes per category, then MaxSim over prototypes instead of over all filings. This is cheaper and less noisy than mean-of-top-3 cosines.
- Validity-aware initialisation transfers too: mask transfers, split rows, refunds or negative amounts and empty payees, which are our "out-of-distribution items".
- Collapse warning: if we learn K free prototype tokens per category or user and associate by argmax, expect winner-takes-all collapse. Initialise from data clusters, or use straight-through Gumbel-Softmax.
- Segment complementarity matches our setup. Explicit (category text, crowd line) helps thin-history users and first-time payees; implicit (history encoders) helps heavy users. With 3.2% overlap their union gains a lot. Our decider-plus-encoder fusion gained only +0.3, which suggests the two make overlapping errors. Measure overlap of correct sets and errors (decider vs MaxSim) by history length before tuning fusion further. A gate on history length or payee familiarity, rather than one global mixing weight, is the analogue of their per-segment wins.
- Does not transfer: ANN budgets, round-robin merging and feed diversity. We need one argmax over a closed set, so their finding that taking the global top across embeddings beats round-robin worse does not apply. The logQ correction for in-batch negatives applies only if a batch mixes households with popular cross-household clusters; for within-household negatives it is not needed.
