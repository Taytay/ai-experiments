# Pinterest engineering posts: summaries, chronological

Each post in this folder summarised (mechanism, evidence, relevance to the transaction categoriser). Written 2026-09-21 from
the fetched markdown by two subagents and the main session (26 Pinterest posts and the Zepto post, dated by publication); numbers are as the posts state them. The applications to our
research are drawn together in `../pinterest-applications.md`.

## 2018-08-15: PinSage: A new graph convolutional neural network for web-scale recommender systems (Ruining He)

Mechanism: A random-walk GCN over the bipartite Pin-board graph ("three billion nodes and 18 billion edges") with visual
and annotation embeddings as input features. Three ideas: on-the-fly convolutions (sample the neighbourhood, build the
computation graph per node instead of the full Laplacian); importance-based neighbourhoods, where the neighbours to
convolve over are "the neighbors with the highest visit counts" from simulated random walks and the aggregator weights them
by importance ("importance pooling"), "a 46% performance gain over the traditional K-hop graph neighborhood method"; and
MapReduce inference (map, join, reduce per aggregation level) for billions of nodes "within a few hours". The stated purpose
of the graph: "by borrowing information from nearby nodes/Pins the resulting embedding of a node becomes more accurate and
more robust. For example, a bed rail Pin might look like a garden fence, but gates and beds are rarely adjacent in the
graph."

Evidence: Pin-to-Pin recommendation against the deployed visual, annotation and combined content embeddings and Pixie:
"outperforms the top baseline by 40% absolute (150% relative)" on recall (MRR likewise); user study: "around 60% of the
preferred items are recommended by PinSage"; A/B in Home Feed and Related Pin Ads "around 30% relative improvement in terms
of user engagement rates"; "25% increase on impressions for our Shop the Look product".

Relevance: The disambiguation sentence is our problem in one line: a merchant string can look like one thing and be
another, and the users who filed it tell you which. PinSage's recipe (content features of the node, plus the content
features of its random-walk neighbours, aggregated by a learned function) is what an encoder-route merchant embedding
should be when the cross-user graph exists: the record text plus the categories and merchants it co-occurs with. In the LLM
route the same information goes into the prompt as a second, collaborative record. PinSage does not model users or
per-user label spaces; it makes the item side better.

## 2018-11-29: An update on Pixie, Pinterest's recommendation system (Pong Eksombatchai)

Mechanism: Pixie is a graph-based recommender that runs random walks over the bipartite Pin-board graph ("more than 175
billion Pins"; full graph "over 100 billion edges", pruned to 20 billion edges = 150 GB held in RAM on graph compilation
machines). From a query node (or a weighted set of query nodes, weighted by the user's action type on each Pin, with more walk
steps given to high-degree query Pins) it walks 100,000 steps, restart probability alpha = 0.5, and ranks nodes by visit
count. Early stopping: stop once "the rank 1,000 candidate gets at least 20 visits" (2x performance). Pruning: cap the
neighbour count of popular Pins, remove edges from Pins mis-saved to the wrong board, and remove "diverse boards (those with
Pins from multiple different ideas)". Supports Pin-to-Pin, Pin-to-Board and Pin-to-Ads; serves "more than 10 billion
recommendations every day" for 250M+ users.

Evidence: "increasing engagement by 37x" versus the pre-Pixie recycle-popular-content era; median saves of served Pins fell
from 90,000 to 1,000 while engagement rose; Pixie "powers over 60 percent of all engagement on Pinterest".

Relevance: The direct analogue is a bipartite graph of merchants and categories (or merchants and users), where a user's
recent transactions are the weighted query nodes and a short random walk with restart yields a "categories co-assigned with
this merchant by similar users" prior. That is a candidate generator for the encoder-prototype route or an extra field in the
merchant fact record, not a replacement for the per-user LLM scorer, since the walk knows nothing about a coined category
name. The board-pruning heuristics (drop "diverse" boards, cap popular nodes like Amazon) transfer as-is.

## 2021-03-05: How Pinterest fights misinformation, hate speech, and self-harm content with machine learning

Mechanism: Pins with the same image are grouped under an image-signature hash; a feed-forward "Pin batch model" scores each
signature over six violation categories plus "safe", using PinSage embeddings (keywords + image, aggregated by a graph
convolution over the Pin-board graph) and OCR text. Training data is "millions of human-reviewed Pins"; daily Spark
inference scores billions of Pins; scores live in RocksDB. A lambda architecture adds an online model of identical
architecture fed by an online PinSage variant that "does not use the Pin-Board graph, so it is less precise" but is
near-real-time (Kafka, Flink, TensorFlow Java). Boards are scored without a board model: a board embedding is the aggregate of
the PinSage embeddings of its most recent Pins, fed into the Pin model; board scores are fanned back to image signatures as
the category-wise average of containing boards' scores.

Evidence: "policy-violating content reports per impression have declined by 52% since the fall of 2019"; "since April 2019,
reports for self-harm content have decreased by 80%".

Relevance: Two small ideas. The board trick: a user's category is a "board", and its embedding can be the aggregate of the
embeddings of the most recent transactions filed under it, which is the encoder-prototype route with recency weighting rather
than all-history averaging. The fan-out: a merchant's prior category distribution can be the average over the categories of
users who filed it.

## 2021-03-30: Detecting Image Similarity in (Near) Real-time Using Apache Flink

Mechanism: A streaming replacement for a batch near-duplicate pipeline that took "several hours"; goal "sub-seconds". Per new
image: LSH terms from visual embeddings; candidates from Manas ranked by matching LSH terms; a TensorFlow classifier against
an empirical threshold; cluster assignment with a random "cluster head" (similarity is not transitive). Scale: 300B Pins,
about 100 images/s (200 at peak), average cluster size 6 but up to 1.1M. Flink stream-stream join synchronises the several
embeddings; tools exist to roll back and to force-fix false positives.

Evidence: latency from hours to sub-second; the post "is not about detecting image similarity but about how to do it in
real-time".

Relevance: Little. The one usable pattern is merchant-string canonicalisation: cluster raw payee strings ("SQ *BLUE BOTTLE
0042") under a canonical head via cheap candidate retrieval followed by a learned verifier, which is how the fact DB's
merchant records would be keyed.

## 2021-05-20: The Evolution of Search at Pinterest (Naveen Gavini)

Mechanism: A product-history post. Guided Search (2014), object recognition and camera search (2015), skin-tone filters. The
one technical remark: "people are taking the same image and putting it on different boards, we can learn the deeper semantics
of an image. We can then train our systems to emulate the ways Pinners are categorizing images."

Evidence: "more than 5 billion searches on Pinterest every month"; Gen Z searches per Pinner up 31% YoY; product searches
"more than 20x YoY".

Relevance: Little, except the quoted thought: the same merchant filed under different category names by different users is a
signal about the merchant's semantics, which is the argument for populating the merchant record with the distribution of
category names other users assign it.

## 2021-07-13: Advertiser Recommendation Systems at Pinterest

Mechanism: Rule- and model-driven suggestions in Ads Manager (bid, budget, expanded targeting) from auction logs; a GBDT
classifier ranks the hundreds of possible recommendations by click probability from advertiser and notification features.

Evidence: none reported.

Relevance: Little.

## 2021-09-09: Pinterest Home Feed Unified Lightweight Scoring: A Two-tower Approach

Mechanism: Home feed candidates come from many generators, "one major source" being Pixie's random walks over the
pin-board graph, each with its own lightweight scorer (a GBDT after the random walk, scoring "thousands of pins' rankings
per request"). These are replaced by one two-tower model: a Pin tower (dense and sparse Pin features through an MLP) and a
user tower (engagement-history features), a dot product between them. Training uses in-batch negatives with a batch of
6,000 "so as to make each positive candidate coupled with enough negative candidates generated for free". Serving: Pin
embeddings precomputed offline for the long-tail corpus, online only for Pins entering "within hours"; user embeddings
computed online per request because "Pinners tend to change their status instantly. For example, if a Pinner engaged with
a cat Pin, they are likely to engage with another cat Pin in the near future."

Evidence: "total saves and closeups both increased 2-3%", "total hides drop 3-4%", plus engineering-velocity and cost
gains.

Relevance: The encoder-prototype route is this two-tower model with the category prototype as the item side and the
transaction as the query, so the recipe applies: in-batch negatives at a large batch, item embeddings cached, the user side
recomputed on every request from the latest history. The reason they give for online user embeddings is the owner's
"hyper-personalised on extremely recent inputs" in Pinterest's words. Their architecture note that a two-tower model
cannot cross user and item features is the reason the LLM prompt (which reads shots and candidate together) beats the
centroid on our set, and why Pinterest keeps the heavy ranker after the two-tower stage.

## 2021-11-09: SearchSage: Learning Search Query Representations at Pinterest (Pancha, Zhai, Rosenberg, Leskovec)

Mechanism: A query embedding for search retrieval and ranking. Two-tower model where the candidate tower is frozen to the
existing 256d fp16 PinSage embedding (so indices and downstream models stay compatible, "at the cost of model performance").
Query tower: distilbert-base-multilingual-cased fine-tuned end to end, a single linear readout on the [CLS] token;
Transformers beat CLSM, bag-of-ngrams and LSTMs "if and only if fine-tuned end to end". Training pairs are (query, engaged
Pin) using only saves and 35s+ click-throughs, no explicit negatives; loss is softmax over in-batch positives, which beat
margin/triplet losses "when mapping into a fixed embedding space". A key heuristic: cap how many times any engaged Pin
appears in training to stop the model learning to always retrieve popular Pins. Eval is Recall@10 against 1M random Pins; a
50/50 organic/shopping blend beat single-task training. Serving batches every 5 ms.

Evidence: +2pp P@8 product relevance weighted by volume (+8pp unweighted), +1pp P@25 for shopping queries, "11% increase in
product long click throughs", 8% more product impressions, "42% increase in related searches".

Relevance: The most directly useful post for the encoder-prototype route. Freezing one tower and learning the other maps onto
keeping the transaction encoder fixed and fine-tuning a small text encoder for category names into the same space; in-batch
softmax on (transaction, category-name) pairs is the right loss; capping per-item positive counts is the analogue of capping
how often "Amazon" or "Groceries" appears so the model does not collapse to head categories. The "only strong engagement"
filter maps to using only user-confirmed (not auto-suggested) assignments as positives.

## 2022-10-26: Query Rewards: Building a Recommendation Feedback Loop During Query Selection

Mechanism: About 30% of Homefeed comes from pin-to-pin retrieval seeded by query pins from the user's history; users may have
"hundreds (or thousands!)" of engaged pins. PinnerSage clusters the user's engaged pins by embedding, each cluster a use case.
Clusters used to be sampled proportionally to raw action counts. Query Reward computes each cluster's engagement rate on the
recommendations it generated and reweights sampling: a large Recipes cluster with impressions and no engagement decays, a
small Furniture cluster that gets engaged rises. Clusters with no reward record get an average weight so they still get
exposure.

Evidence: no numbers reported.

Relevance: Moderate, for shot selection. The 24 shots are the categoriser's "query pins"; the user's history can be clustered
(by merchant embedding or by category) and shots sampled across clusters for coverage, with per-cluster weights updated by
whether shots from that cluster led to accepted predictions. The "average weight for clusters with no record" rule is a cheap
exploration fix for rarely used categories that otherwise never appear in the shots.

## 2022-11-04: How Pinterest Leverages Realtime User Actions in Recommendation to Boost Homefeed Engagement Volume (Xia, Gu, Badani, Zhai)

Mechanism: The Pinnability Homefeed ranker gains a realtime feature: the user's latest 100 actions with GraphSage pin
embeddings, action type (repin, click, hide) and timestamp, complementing the static long-term PinnerFormer embedding. A
transformer encoder consumes a stacked [candidate_pin_emb, action_emb, engaged_pin_emb] matrix (early fusion of the candidate
with every history item, "the majority of engagement gain"); a random time-window mask hides some actions within one day of
request time "to make the model less responsive and to avoid diversity drop". v1.1 added layers and kept the first 10 output
tokens plus a max-pooled token. Engagement decayed unless retrained, so the model is retrained twice a week. CPU latency rose
"more than 20x", forcing GPU serving.

Evidence: offline "+8.87% offline repin and a -13.49% hide drop" in HIT@3, beating average pooling, CNN, RNN, LSTM and a
vanilla transformer; online A/B on 1.5% of traffic: repin volume +6% overall, +11% for non-core (new, casual, resurrected)
users, hide volume -10%; v1.1 a further 5% repin.

Relevance: High, conceptually. The 24-shot prompt is already a "last N actions" sequence; this post argues for (a) early
fusion, each shot scored in relation to the candidate transaction (the LLM does this implicitly; the prototype route compares
the candidate to a mean), (b) including the action type, marking shots the user corrected versus accepted, and (c) the
biggest lift landing on low-history users, which matches the cold-start users the categoriser worries about. The
retrain-twice-weekly finding warns that recency-driven models drift.

## 2023-05-02: Large-scale User Sequences at Pinterest

Mechanism: Platform infrastructure behind the realtime-actions post. Lambda architecture: Flink reads Kafka events, enriches
each, and appends to a RocksDB wide-column store, one dataset per event type (e.g. last 10 repins but last 100 close-ups).
Store requirements: O(1) inserts, out-of-order inserts kept in reverse chronological order, store-side deduplication. A Spark
pipeline corrects enrichments, bootstraps new event types, and is where "most relevant N events instead of last N" would be
tried; it stores "the last few thousand events for each user". Target latency "< 2 seconds" from action to response.

Evidence: "significant user engagement gains" in Homefeed (no numbers).

Relevance: Little for modelling. The explicit note that "last N" is a placeholder for a selection rule and "most relevant N"
is the intended successor is the retrieved-shots question already queued (REAL-9). Per-event-type caps suggest capping shots
per category so one dominant category does not fill all 24 slots.

## 2023-05-09: An ML based approach to proactive advertiser churn prevention

Mechanism: A GBDT predicts advertiser churn over 14 days from 200+ windowed features; SHAP contributions give churn drivers.

Evidence: online AUC within 1 to 3% of offline; "24% reduction in the churn rate of high tier pods".

Relevance: Little. Methodological only: report offline-vs-online agreement, and pick operating thresholds from the consumer's
stated precision and recall needs (when to auto-apply a category versus ask the user).

## 2023-06-13: Deep Multi-task Learning and Real-time Personalization for Closeup Recommendations

Mechanism: The Closeup (Related Pins) ranking DNN outputs per-action likelihoods (repin, closeup, click, long-click) rather
than one score. Representation layer: a transformer encoder over the "User's most recent 100 engagement actions (repin,
closeup, hide, etc.)" and "most recent 100 engaged pins' pinSage embeddings", plus query-Pin, Pinner and candidate-Pin
embeddings; then a summarization MLP per feature group, a "transformer mixer" (self-attention over groups), and an MMoE head.
Score regularization distils from the previous production model to stop rank flapping between retrains. A "Learned Utility"
blending model learns per-request weights over the task predictions from randomised blender weights and a reward
(closeup = 1, hide = -2).

Evidence: MMoE online A/B: "repin volume increased by 4% and closeup volume increased by 1%".

Relevance: The transferable part is the input recipe: a fixed-length window of the user's most recent N actions, each carrying
an action type and a content embedding, fed as a sequence. That is the shape of the 24 recent labelled shots, and it argues for
ordering shots by recency and carrying the label as a token per shot. MMoE and Learned Utility solve a multi-objective
blending problem the categoriser does not have.

## 2023-09-05: MLEnv: Standardizing ML at Pinterest Under One ML Engine

Mechanism: Platform post. "10+ different ML frameworks" in 2021 replaced by one engine (monorepo, one Docker runtime, GPU
unit tests, MLFlow, distributed and mixed-precision training, GPU serving) with "no abstraction to the modeling logic";
adoption "from <5% in 2021" to "95% of ML jobs". The Home feed team's "Realtime User Actions Sequence" with transformer and
DCNv2 propagated to other surfaces "within days" once the stack was unified.

Evidence: "300% increase in the number of training jobs"; engagement gains "on the order of mid-double digit percentages"
attributed to the paradigm.

Relevance: Little, beyond confirming that the real-time user-action sequence plus transformer was the model change worth
copying across surfaces in 2022.

## 2023-09-12: Last Mile Data Processing with Ray

Mechanism: Dataset iteration through Spark and Airflow took "several weeks" per variation; Ray Data's streaming execution
transforms data concurrently with training on a heterogeneous CPU+GPU cluster.

Evidence: "20% improvement in the training throughput" from the loader alone, "up to 45% faster" with filtering and dynamic
negative downsampling, 90 hours to 15 hours end to end.

Relevance: Little. Keep sampling and negative-selection logic in the loader so a new variant (which 24 shots, how the encoder
route draws negatives) is a config change.

## 2023-09-26: Training Foundation Improvements for Closeup Recommendation Ranker

Mechanism: Hybrid logging (a low share of impressions, all positive engagements, and a small slice of "completely randomly
ordered candidates" logged in full for offline replay, calibration and evaluation); a configurable PySpark sampler with
lineage caching; an Auto-Retraining Framework with validation against the previous production model, which is also the
distillation teacher.

Evidence: "model refreshes bring in better performance across all refresh cadence" (daily to bi-weekly); weekly chosen; ARF
onboarding "from 3+ hours to 30 minutes".

Relevance: Two things. The randomised slice argues for an unbiased evaluation subset whose shots are drawn uniformly rather
than by recency or retrieval, so the shot-selection policy can be replayed offline. The refresh-cadence finding is the
recommender version of "hyper-personalised on recent inputs": drift is real and fresh data always helped, which supports the
time-axis rows (REAL-9, REAL-11).

## 2024-03-20: LinkSage: GNN-based Pinterest Off-site Content Understanding

Mechanism: A GNN embedding for landing pages. Training pairs: (Pin, landing page) labelled positive when the Cohesion
signal exceeds a threshold; negatives in-batch plus random. Graph: a heterogeneous (Pin, landing page) graph built with
XPixie, Neardup image clustering to densify, pruning so popular nodes do not dominate; random walks give each landing page "a
list of its neighbor landing pages and their visit counts". Features: self text embedding, size-weighted crawled-image
embeddings, the same for neighbours, plus "graph node visit counts and self degree". Model: a transformer encoder over self,
neighbour (sorted by visit count, with position embeddings) and structure tokens. Matryoshka training gives five embedding
sizes from one model; PinSage is the Pin-side target so the space is XSage-compatible; incremental serving re-embeds only
daily-crawled pages.

Evidence: Recall@100 with 256 dims 72.9%; 64 dims "saves 75% of the cost and the performance only drops by 8.3%". Online
"significant gains" on repin, long click, engaged sessions, hide, Ads conversions and Board Picker (tables only).

Relevance: The closest analogue to the merchant fact DB, and the clearest statement of what the "graph" does at the model
level: the random walk chooses each node's neighbours, and a transformer reads the node's own features beside its neighbours'
features. For the LLM route that suggests appending "similar merchants and how this user labelled them" to the record; for
the encoder route it is a feature recipe (neighbour embeddings, visit counts, degree). Users are not nodes here, so it does
not address per-user label spaces.

## 2025-06-06: Next-Level Personalization: How 16k+ Lifelong User Actions Supercharge Pinterest's Recommendations (TransAct V2)

Mechanism: TransAct (2023) modelled the last 100 actions in the home feed ranker; V2 models "up to 16,000 user actions: a
160x scale-up", each action a token of timestamp, action type, surface and a 32-d PinSage embedding (int8-quantized). The
key step is nearest-neighbour selection at ranking time: for each candidate pin the model receives "the most recent r
actions" plus "the top K nearest neighbors from the lifelong, real-time, and impression sequences, based on dot-product
similarity in PinSage embedding space", concatenated. Early fusion (candidate embedding concatenated to each action), a
two-layer causal transformer of dimension 64, max pooling; an auxiliary Next Action Loss (predict the next engaged pin
against sampled negatives), where "impression-based negative samples (pins shown but not engaged) are much more effective
than random negative sampling". Serving: NN features logged instead of the full sequence (O(1) not O(L)), on-device NN
search per request, a fused Triton kernel (6.6x over Flash Attention at this size), request-level deduplication (8x less
PCIe transfer); "75-81% lower p99 model run latency".

Evidence: offline "+13.31% Top-3 Repin Hit ... more than 2x improvement over previous systems", "-11.25% Top-3 Hide Hit";
online on 1.5% of traffic "6.35% Repin increase", "12.8% fewer Hides", "+1.41% time spent", where "previous model launches
over the past two years typically achieved lifts around 0.2-1%".

Relevance: This is PLAN row 41 (REAL-9) as Pinterest built it: the shots a candidate sees are the most recent r plus the K
history rows nearest to the candidate, selected per candidate, and the two rules are combined rather than chosen between.
The impression-negative finding maps to training with the categories the user saw and did not choose (the other names in
the scheme) as the negatives, which the label SFT already does implicitly and the encoder route should do explicitly. The
scale of the gain over the last-100 window is the strongest evidence in the set that history selection, not model size, is
where a per-user recommender's accuracy comes from.

## 2026-04-13: Scaling Recommendation Systems with Request-Level Deduplication

Mechanism: The Foundation Model has "a 100x increase in transformer dense parameter counts"; request-level data is
"approximately 16K tokens encoding all actions a user has taken", duplicated "hundreds to thousands of copies per request"
across scored items. Storage sorted by user and request ("10-50x storage compression"); request-sorted batches broke IID
(BatchNorm, fixed with SyncBatchNorm); in-batch negatives became false negatives ("~0% ... to as high as ~30%"), fixed by
taking negatives only from other users. The user tower runs once per request; for ranking, DCAT runs the transformer over the
user history once, caches K/V per layer, and lets each candidate cross-attend.

Evidence: "4x end-to-end training speedup for retrieval", "~2.8x speedup for ranking", "7x increase in ranking serving
throughput".

Relevance: Directly applicable to the scorer's cost structure: the 24 shots plus record are the "request", the user's category
names are the "items"; encode the shared prefix once and let each candidate label attend to it (KV-cache reuse), which the
batched Scorer approximates. The user-level negative masking transfers verbatim to any contrastive training of the encoder
route: never draw a negative from the same user.

## 2026-05-01: Optimizing ML Workload Network Efficiency (Part I): Feature Trimmer

Mechanism: Root-leaf online serving; the root fetches features and fans scoring requests out to leaf GPU partitions; the
Feature Trimmer removes features a leaf's models do not consume before the fan-out, cutting network volume.

Evidence: infrastructure savings (not read in detail).

Relevance: Little.

## 2026-05-21: Making User-Sequence Data More Cost-Efficient, Faster, and Easier to Use

Mechanism: A redesigned user-sequence platform. A sequence is "an ordered list of recent, relevant events for a user, along
with the enrichments"; example "the last 500 engagements a user had with Pinterest Pins". Quality dimensions: freshness,
completeness, consistent enrichment between batch and streaming, stable versioned schemas. "Define a signal or event type
once, then instantiate it consistently across multiple runtimes." A serving API answers "Request sequence X for user U" with
trimming such as "last N events within this time window"; streaming is the "now" view, batch "fixes history".

Evidence: qualitative by policy: cost reductions, faster onboarding, "improved engagement metrics".

Relevance: A specification for what the 24 shots should be: an enriched, recency-ordered window with a documented freshness
profile, built by the same code at training and scoring time so the two never drift. The training/serving split-brain
warning applies when REAL-9 builds retrieved and recent shots by one path and the eval harness by another.

## 2026-06-22 (other): Real-Time Personalisation at Scale: How Zepto Understands What You Want, Right Now

Mechanism: A quick-commerce ranker replacing collaborative filtering and cohort models that had "no session awareness",
"time blindness" and a "static user representation". A dual-sequence reranker after DIN and BST: each interaction is a
token of item embedding (128-d), action embedding ("an Add-to-Cart carries heavier intent than a mere view") and a
temporal encoding of the exact hours elapsed through a small MLP; two separate transformer encoders for long-term history
and the current session; target-aware pooling where the candidate SKU is the query over the sequence ("the user profile is
dynamically recomputed per candidate"); a learned fusion gate between session and history ("an empty session relies
entirely on history"); a ranking head with per-user counters, 28-day and 2-hour popularity windows and calendar context;
a hybrid loss (in-session listwise, batch-sampled softmax negatives from other users, and a repeat-count-weighted BCE for
replenishment).

Evidence: "Conversion Lift, Tail Item Discovery and Lightning Fast Serving (P99 in low single digit ms)"; no numbers in the
text read.

Relevance: The closest published system to the owner's framing, from a consumer app rather than a content platform. Its
three criticisms of collaborative filtering are the case against a static per-user model of any kind; the gate between a
long history and the last few events is the mechanism row 43's relabelling events test; the temporal encoding of elapsed
time is a cheap addition to each shot ("3 days ago"); and the repeat-count weighting is the transaction analogue exactly
(a merchant the user has filed 40 times is a replenishment, not a discovery).

## 2026-06-25: Achieving Near-Linear Training Scalability for Pinterest's Foundation Models

Mechanism: Multi-node training of the Foundation Ranking Model ("approximately 99% of parameters reside in embedding
tables"); FP8 communications, balanced sharding, 2D parallelism; torch.compile gave "a 55% single-node throughput gain".

Evidence: "7.5x scaling factor at 8 nodes (64 GPUs), 93.75% of ideal"; larger models gave "statistically significant
engagement gains", enabling teacher-student distillation.

Relevance: Little. A bigger teacher distilled into a smaller serving model is how Pinterest cashes in scale.

## 2026-08-26: Scaling Conditional Learned Retrieval for Pinterest Home Feed

Mechanism: CLR conditions the user tower on an explicit context, producing several condition-aware user embeddings per
request. Conditions: taxonomy interests (later an LLM-generated interest signal); Pin conditions from clustering recently
engaged Pins (medoids); Board conditions from random walks over the Pin-Board graph. Pin tower: a 20GB pretrained ID table
plus semantic IDs from residual-quantized VAEs ("five hierarchical layers, each layer has 2048 possible codes") so long-tail
Pins "borrow" collaborative signal. User tower: a Conditioned User Sequence Transformer appends condition tokens to the
action sequence ("The condition token acts like a query over recent user actions"); later PinFM fine-tuned inside the tower.
"M-Falcon": all of a user's conditions flattened onto one sequence with a block mask, "2x" throughput.

Evidence: Board and Pin conditions gave "large metric wins" while deprecating heuristic candidate generators; serving changes
"reduced p90 model latency by 85% (80ms to 12ms)".

Relevance: The most transferable post. A condition token appended to the recent sequence, "acting like a query", is a
candidate category name scored against the 24 shots; M-Falcon's block mask (conditions attend to history, not to each other)
is how to score all of a user's category names in one forward pass on either route. Semantic IDs are an answer to cold-start
merchants: hierarchical codes from the merchant record let an unseen merchant share rows with similar ones. Random-walk Board
conditions map to "categories that this user's neighbours in the merchant graph tend to use", a candidate source for coined
names.

## 2026-09-11: Evolving Pinterest's Embedding Retrieval Platform

Mechanism: Manas ("over 80 clusters", "billions of embeddings", HNSW/IVF, real-time updates within seconds): product and
scalar quantization (SQ at "recall over 90% consistently", index 59 to 75% smaller), SSD serving with SPANN+PQ ("over 40% of
CPU time" saved at "<5% recall drop"), and multi-embedding retrieval with ColBERT-style "Sum of MaxSim".

Evidence: "over 50% memory reduction in embedding indices and 20-30% cost savings".

Relevance: Little for the model. The pointer is late interaction: represent a transaction and a category as several vectors
and score with Sum of MaxSim rather than one prototype dot product, which handles category names that match only part of a
transaction's text (the FastFit idea, row 40).
