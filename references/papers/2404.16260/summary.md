# OmniSearchSage: Multi-Task Multi-Entity Embeddings for Pinterest Search

- arXiv 2404.16260 (v1, 25 Apr 2024) - https://arxiv.org/abs/2404.16260
- Prabhat Agarwal, Minhazul Islam Sk, Nikil Pancha, Kurchi Subhra Hazra, Jiajing Xu, Chuck Rosenberg (Pinterest)
- Venue: TheWebConf (WWW) 2024 Industry Track, oral (Companion Proceedings, pp. 121-130, DOI 10.1145/3589335.3648309)
- Source: references/papers/2404.16260/paper_flat.tex

## One-paragraph summary
OmniSearchSage replaces SearchSage at Pinterest with one query embedding trained jointly against three kinds of target: Pins, products and related queries. All share one 256-d space. The query tower is multilingual DistilBERT with CLS, a projection to 256 and L2 normalisation. The Pin and product tower is a cheap unified encoder: three tokenisers (word unigram, word bigram, character trigram), a 2-hash hash-embedding table, summed token embeddings concatenated with continuous features (PinSage, unified image embedding, ItemSage), and a 3-layer MLP. Training is multi-task sampled softmax with in-batch and random negatives and logQ correction. The main gains come from document enrichment: BLIP captions for Pins without text, the titles of boards users saved the Pin to, and queries that led to engagement with the Pin. Extra compatibility encoders keep the query embedding usable against the frozen legacy PinSage and ItemSage embeddings at no cost. Offline recall@10 rises 26 to 87% over SearchSage (Table 2). Online: +7.4% search fulfilment, +3.5% relevance, and up to +5.27% ads gCTR. The query embedding is served at 300k QPS behind a 30-day-TTL cache.

## Problem
Pinterest search has to retrieve and rank Pins, products and query suggestions in 45+ languages. Separate query embeddings per content type multiply models, inference cost and maintenance, and low-label tasks learn poor representations alone. About 30% of Pins lack a usable title or description. New embeddings must also stay compatible with existing ones that downstream systems depend on.

## Method
- Entities: query, Pin, product. Encoders (Section 3.4):
  - Query encoder: distilbert-base-multilingual-cased, CLS, linear to 256-d, L2 normalisation (Figure 2).
  - Unified Pin and product encoder (Section 3.4.2, Figure 3). It is deliberately simple so batches, and therefore in-batch negative counts, can be large.
    - Tokenisers: word unigrams (200k vocabulary), word bigrams (1M), character trigrams (64k). Out-of-vocabulary tokens are dropped.
    - Hash embedding: a table of 100,000 rows, 2 hashes per token. Token embedding = W_1i h_1(i) + W_2i h_2(i), with learned per-token weights.
    - The sum of token embeddings is concatenated with continuous features (PinSage, unified visual embedding, ItemSage for products; missing features set to zero) and fed to an MLP of 1024, 1024, 256, then L2 normalisation.
  - Compatibility encoders (Section 3.4.3): separate Pin and product towers that take the frozen PinSage and ItemSage embeddings as inputs, so the query embedding is also trained to match legacy embeddings.
- Text enrichment (Section 3.2). Coverage: 71% of entities have a title or description, 91% have board titles, 65% have engaged queries, 100% have captions.
  - Synthetic captions: BLIP captions for all Pins and products. In human evaluation of 10k images with 3 raters each, 87.84% were relevant and high quality and 1.16% irrelevant and poor.
  - Board titles: titles of all boards the Pin was saved to, up to 10 unique ones.
    - Each title is scored by its frequency and the prevalence of its words.
    - Titles are ranked by score (ascending), then word count (descending), then character length (descending), which removes noisy and redundant titles.
  - Engaged queries: queries that led to engagement with the Pin, sorted by a function of engagement counts and types; the top 20 are kept.
    - A two-year window beat shorter ones.
    - The list is updated incrementally every n days by merging new logs into the previous top 20.
- Loss (Section 3.5): extreme classification with sampled softmax (Eqs. 1-5). Each task T = {dataset D of (query, entity) pairs, entity encoder E}.
  - L_T = L^{bn} (in-batch positives as negatives) + L^{rn} (a random catalogue sample C' as negatives), both with logQ correction: subtract log Q(y|x), or log Q_n(y) for independent random negatives.
  - Total L = sum over tasks. Task influence is set by the mix of tasks in each batch, and pairs are shared across tasks that use the same dataset.
- Data (Table 1), one year of logs:
  - Query-Pin: 1.5B pairs (repin, long click).
  - Query-Product: 136M from query logs plus 2.5M offsite pairs (add-to-cart, checkout).
  - Query-Query: 195M pairs (clicks on related-query suggestions).
  - Popularity cap: at most 50 pairs per Pin and 200 per product.
- Serving (Section 3.6):
  - The query encoder runs on GPU in a C++ model server, behind a cache with a 30-day TTL, because query frequency follows Zipf's law.
  - 300k requests/s, p50 latency 3 ms, p90 20 ms; the inference server sees about 500 QPS.
  - Pin and product embeddings are computed by daily offline GPU batch inference.
- Evaluation: Recall@10 of the engaged entity against 1.5M uniformly random corpus entities. There is a 15-day gap between training and the 7-day evaluation window (the engaged-query features could otherwise leak). The sample is 80k pairs per type, and human-labelled relevance pairs come from US, UK, FR and DE. (The Recall@10 formula is printed with the inequality reversed.)

## Experiments and results
- Against SearchSage, which used frozen PinSage and ItemSage (Table 2), Recall@10:
  - Pin save 0.39 to 0.65 (+67%); long click 0.45 to 0.73 (+62%).
  - Pin relevance: US 0.25 to 0.45 (+80%), UK 0.29 to 0.51 (+76%), FR 0.23 to 0.43 (+87%), DE 0.28 to 0.46 (+64%).
  - Product save 0.57 to 0.73 (+28%); long click 0.58 to 0.73 (+26%).
  - Query click 0.54 to 0.78 (+44%). SearchSage was never trained on query-query pairs.
- Captions, on the 24k evaluation pairs whose Pin had no title or description (Table 3): save 0.51 to 0.66 (+30.4%), long click 0.60 to 0.76 (+25.6%), relevance 0.36 to 0.36 (+0%).
- Cumulative enrichment (Table 4), save / long click / relevance:
  - Continuous features only: 0.43 / 0.53 / 0.30.
  - Plus title, description and captions: 0.52 / 0.63 / 0.39 (+21 / +19 / +30%).
  - Plus board titles: 0.61 / 0.68 / 0.44 (+17 / +8 / +13%).
  - Plus engaged queries: 0.65 / 0.73 / 0.46 (+7 / +7 / +5%).
- Multi-task against single-task models with equal batch size, compute and iterations (Table 5):
  - Pin save 0.68 single vs 0.65 multi; long click 0.75 vs 0.73; relevance 0.45 vs 0.46.
  - Product 0.73 / 0.73 in both.
  - Query click 0.73 single vs 0.78 multi.
  - Low-data tasks gain or stay level; the 1.5B-pair task loses slightly.
- Compatibility encoders (Table 6): the jointly trained encoders match the separately trained SearchSage (Pin save 0.39 vs 0.39, long click 0.45 vs 0.43, relevance 0.26 vs 0.26; product 0.57 / 0.58 vs 0.57 / 0.57). The learned encoder shows "almost no noticeable degradation".
- Human evaluation: 300 queries (head and tail), top 8 Pins, 3 judges each, agreement 0.89. Relevance is +10% over token-based retrieval.
- Online organic search (Table 7), fulfilment / relevance:
  - Retrieval: +4.1% / +0.5%.
  - L1 scoring: +0.5% / 0.0%.
  - L2 scoring and relevance model: +2.8% / +3.0%.
  - Cumulative: +7.4% fulfilment, +3.5% relevance.
- Ads (Table 8), gCTR: product ads retrieval +5.27%, ads engagement model +2.96%, ads relevance model +1.55%. Ads relevance +4.95%.
- Query interest classification over a hierarchical taxonomy: precision +30% on average over a FastText baseline, with larger gains at finer levels.

## Limitations
- There is no ablation of the tokeniser or hash-embedding choices, of the number of random negatives, or of logQ; the paper says only that "numerous ablation studies" led to the design.
- The text-enrichment ablation is cumulative in one fixed order, so the marginal value of each source in another order is unknown.
- Board titles and engaged queries need engagement history, so they are absent for new Pins (91% and 65% coverage). Captions fill only part of that gap.
- Recall is measured against random negatives, and online results are relative lifts without confidence intervals.
- The query embedding is not personalised: no user context.

## Relevance to this workspace
- Board titles are our crowd line, as an encoder feature rather than prompt text.
  - A board title is a user-written name for a collection the item was put in. That is exactly "other households filed this payee as: X".
  - On top of the item's own text, board titles were worth +17% save and +13% relevance; engaged queries added a further +7%.
  - Try: enrich the payee side of the history-aware encoder (bge-small) with the top 10 crowd category names for that bank string. Deduplicate and rank them as the paper does (frequency score, then longer and more specific names first). Train and score with the enriched text, so the encoder, not only decider-4B, sees cross-household labels.
  - Expected effect: largest on first-time payees, where only content and crowd labels exist.
- Synthetic captions correspond to an LLM-written gloss of a cryptic bank string ("SQ *JOES 0423" becomes "coffee shop / café"). This helped only where text was missing (+30% engagement recall on text-less Pins) and did nothing for relevance. Spend it only on strings the encoder cannot read: low-information strings and processor prefixes, not clean merchant names. (PLAN row 184's description line is the prompt-side version.)
- The cheap tower (unigram + bigram + character-trigram hash embeddings feeding an MLP to 256-d) is a strong, fast baseline for noisy bank strings.
  - Its point was large batches and therefore many in-batch negatives.
  - Worth one arm as a payee encoder, alone or concatenated with bge-small, in the kNN/MaxSim pipeline.
- One query tower trained against several target types, with low-data tasks gaining (Table 5), supports training one transaction encoder jointly on:
  - transaction to earlier transaction (current InfoNCE);
  - transaction to category-name text (the TransAct-style query);
  - payee to cross-household cluster (the planned graph targets).
  - Expect the biggest task to lose a little.
- Sampled softmax with both in-batch and random negatives, plus logQ correction, and a cap on pairs per popular item (50 per Pin) are cheap changes to our InfoNCE:
  - cap the training pairs per frequent payee or category so "Groceries" does not dominate;
  - subtract log frequency from in-batch logits.
- Compatibility encoders show a new encoder can be co-trained to stay compatible with a frozen older embedding at almost no cost. Useful if a stored index of past transaction embeddings must survive an encoder upgrade, which matters at more than 1M users.
- A query-embedding cache with a 30-day TTL (Zipf distribution): bank strings are Zipfian too. Cache payee embeddings per normalised string; most real-time categorisations then need no encoder call.
- Does not transfer:
  - multilingual DistilBERT;
  - the image features and PinSage inputs;
  - query-query pairs;
  - the GPU serving stack at 300k QPS.
  - The query embedding is user-agnostic, while our labels are personal; the per-household history layer still has to sit on top.
