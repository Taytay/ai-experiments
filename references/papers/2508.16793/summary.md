# Bootstrapping Conditional Retrieval for User-to-Item Recommendations

- arXiv 2508.16793 (v1, 22 Aug 2025) - https://arxiv.org/abs/2508.16793
- Hongtao Lin, Haoyu Chen (equal contribution), Jaewon Yang, Jiajing Xu (Pinterest). meta.json spells the third author "Jaewon Jang"; the tex says Yang.
- Venue: RecSys '24 (18th ACM Conference on Recommender Systems, Bari), a short or industry-talk paper with a speaker bio; DOI 10.1145/3640457.3688057
- Source: references/papers/2508.16793/paper_flat.tex

## One-paragraph summary
Conditional retrieval means finding items that are relevant to both a user and a given condition, such as a topic. The authors train such a retriever on ordinary (user, engaged item) pairs, with no logged condition. They take a condition from the positive item's own metadata (a topic sampled at random from the item's item-to-topic set) and feed it into the user tower's embedding layer, so it crosses with the user features in the feature-crossing layers. Everything else is a standard two-tower model with in-batch sampled softmax. They tested it on a new topic-based notification feed (Table 1). Without a filter, the conditional model (CR) matches the topic 82.8% of the time, against 20.3% for the plain two-tower model (LR). CR beats LR-plus-filter on email CTR (+2.86% vs +1.87%) and push CTR, and costs +180k a year to serve instead of +1.2M. CR with a filter was shipped and gave +0.26% weekly active users.

## Problem
New condition types (topic, brand, colour, merchant) keep appearing, and none of them come with condition-specific engagement data. The usual workaround is a two-tower model followed by over-fetching and post-filtering, or filtering during ANN search. That ignores condition relevance during training, and it gets expensive when the condition is rare among the user's nearest items. Training on (user, condition) pairs from logs needs a product that already exists and logs the condition, which is not true when bootstrapping a new surface. The task also has two objectives: engagement and relevance to the condition.

## Method
- Two-tower base (Fig. 1): user and item features each pass through an embedding layer and feature-crossing layers (e.g. MLPs). The score is the dot product of user and item embeddings, served by ANN (HNSW).
- Condition Extraction Module: takes item metadata and produces a single condition. It is built independently of the two-tower model. For topics it uses an existing item-to-topic feature and samples one topic at random from the positive item's topics.
- Conditional User Tower: the condition embedding enters at the embedding layer of the user tower and goes through the same feature-crossing layers. The paper argues this enables "higher order feature interactions between user and condition". The item tower is unchanged.
- Training is the same as a standard two tower: engaged user-item pairs are positives, other items in the batch are negatives, and the dot product feeds a sampled softmax (mixed-negative sampling citation). Training data is identical to LR.
- Serving: the user tower is run with each target topic as the condition, then ANN search. Optionally a streaming filter (mini-batched ANN plus filter until enough items arrive or time runs out).
- Baselines: INDEX (topic to items, top-k by popularity, not personalised), and LR (plain two tower trained on the existing notification feed, plus topic filter).
- Future work, stated: hard negatives with the same condition inside the sampled softmax, and an auxiliary loss between the conditional user embedding and the target condition to raise relevance.

## Experiments and results
- Online A/B test only (Table 1). Each arm retrieves a few hundred items and about 20 survive ranking; total send volume is fixed. Gains are relative to a control without the new notification type:
  - INDEX: email CTR +1.38%, push +1.25%, WAU +0.10%, infra +40k
  - LR with filter: +1.87%, +2.15%, WAU +0.23%, infra +1.2M
  - CR without filter: +2.86%, +2.63%, WAU +0.27%, infra +180k
  - CR with filter: +2.94%, +2.58%, WAU +0.26%, infra +300k
- Topic match rate without a filter: LR 20.3%, CR 82.8%. This is why LR's filter is so costly: it must page through about four times more ANN candidates.
- For CR the filter is neutral on engagement. It guarantees relevance but drops "somewhat relevant but highly engaging" items. CR with filter shipped so that relevance is guaranteed.
- No offline metrics, no ablations, no significance values.

## Limitations
- Two pages long, online results only. There is no offline recall table, no ablation of where the condition is injected or how it is sampled, and no confidence intervals.
- Sampling the condition from the item's attributes can disagree with the user's real intent at serving time. The follow-up paper (2506.23060, Table 3) shows that logging the condition at engagement time beats this.
- The condition vocabulary is a fixed global topic set with a learned embedding table, so conditions that are open or unseen are not covered.
- Same-condition hard negatives are left for future work. In-batch negatives mostly carry other topics, so the model can learn "is this item about topic t" more than "which item on topic t does this user want".

## Relevance to this workspace
- The mapping is direct if we turn the problem around. The condition is a category, the item is a transaction, and the user tower holds the user state. Score each of the user's categories c as dot(phi(u, c), psi(txn)) and choose the argmax. This is a per-category query, unlike MaxSim (category as a set of its filings) and unlike the decider (one slot per category in one prompt). What CR adds is that user and category are crossed before the dot product. That suits personal purposes (a parent's medication, a named trip), whose meaning depends on the household.
- Condition extraction comes free: every filed transaction already carries its category (the "source interest", which the follow-up paper found better than inferred conditions). Our categories are per-user names, not a global ID table, so the condition embedding must be computed, not looked up. Use bge text of the category name, the mean of the category's earlier filings (a prototype), and optionally the cross-household cluster id (one of 64) as a learned ID embedding. That cluster id is the one thing a global table can hold.
- The planned row 196 (user tower plus FiLM or additive condition tokens per category) is this paper's architecture. The paper injects at the embedding layer and crosses with MLP or DHEN, which is closer to concatenate-then-MLP than to FiLM. Try both on the same encoder.
- Negatives: the paper's stated next step, same-condition hard negatives, is in our case the user's other transactions and the user's other categories. In-batch negatives across households are too easy; the 82.8% topic-relevance figure shows how much the condition alone does. Use within-household batches.
- Bootstrapping argument: because training needs only (txn, filed category) pairs, it extends to categories nobody has seen, as long as the condition encoder generalises from the name text. This matters for first filings into a newly created category, where kNN and MaxSim have no data.
- Does not transfer: ANN serving cost and the streaming filter. We have at most about 100 categories per user and score all of them exhaustively. Relevance filtering corresponds to restricting to the user's own category list, which we already do.
