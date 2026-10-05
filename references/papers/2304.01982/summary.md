# Rethinking the Role of Token Retrieval in Multi-Vector Retrieval (XTR)

- arXiv 2304.01982 (v3, 2024-04-08; v1 2023-04-04) - https://arxiv.org/abs/2304.01982
- Jinhyuk Lee, Zhuyun Dai, Sai Meher Karthik Duddu, Tao Lei, Iftekhar Naim, Ming-Wei Chang, Vincent Y. Zhao (Google DeepMind)
- Venue: NeurIPS 2023
- Source: references/papers/2304.01982/paper_flat.tex (read 2026-10-05 for PLAN rows 209-210)

## One-paragraph summary
ColBERT-style models are trained on document-level sum-of-max but served through token retrieval; training never penalises a negative document's very high token scores once the document softmax is right (App. A: dL/dP- = (1/n) P(D-|Q)), so many irrelevant tokens score high and token retrieval misses gold tokens. XTR trains with an in-batch top-k alignment (a document's MaxSim counts only query-token matches in the top-k_train of the batch's tokens, normalised by the number of query tokens that matched, Z) and at inference scores candidates only from retrieved token scores, imputing missing ones with the k'-th retrieved score: ~4000x fewer scoring FLOPs. XTR_xxl 52.7 BEIR nDCG@10 vs T5-ColBERT_xxl 50.8 and GTR_xxl 49.1, without distillation.

## Method
- f_ColBERT = (1/n) sum_i max_j q_i.d_j (the 1/n "stabilizes training"); in-batch CE.
- Training: A_ij = 1[j in top-k_train over all batch tokens]; f_XTR = (1/Z) sum_i max_j A_ij q_i.d_j, Z = query tokens with >= 1 retrieved token of D (Z better than n).
- Inference: top-k' tokens per query token (ScaNN); f = (1/n) sum_i max_j [A_ij q.d + (1-A_ij) m_i], m_i = the k'-th retrieved score. Training directly with the imputed score does not converge.
- T5 encoders (base, xxl), MS MARCO with RocketQA hard negatives, 50k iterations, lr 1e-3, batch 128-320, k_train in {32..320}, k' = 40,000; d = 128 in the FLOPs setting; no temperature.

## Experiments and results
- BEIR nDCG@10 (13 sets): XTR_base 49.1 vs T5-ColBERT_base 46.8, GTR_base 45.2, ColBERT 45.1, BM25 44.0; distilled ColBERTv2 49.9; XTR_xxl 52.7. MS MARCO in domain XTR_base 45.0 vs T5-ColBERT 45.6. ArguAna (long queries): XTR_base 40.7 vs GTR_base 51.1 (multi-vector weak on long queries).
- BEIR recall@100: 68.0 vs 65.5 (base). EntityQuestions top-20: XTR_xxl 79.4 vs GTR_xxl 75.3. MIRACL: mXTR_base 52.2 vs mContriever 41.5 (English-only training).
- Imputation (MS MARCO dev MRR@10): XTR none 22.6, m=0 36.2, m=0.2 36.4, k'-th score 37.4; T5-ColBERT with the cheap scorer 0.0 without imputation.
- Smaller k_train better at small k'; larger batches better; XTR less lexical ("usual" -> finds "average").

## Limitations
MS MARCO only; k_train optimum depends on batch and data; efficiency argued in FLOPs; no calibration; TPU scale.

## Relevance to this workspace
- Imputation: today's ColBERT reader scores only categories with a filing among the 50 nearest earlier transactions; where a cut stays, impute the cut-off score rather than 0 or dropping the category. Row 210 avoids the cut: every option is scored in full.
- Train under the same cut as inference, or drop the cut (row 210 does).
- Sum-of-max CE leaves high-scoring negative tokens alone: a calibration risk for row 210's softmax that the Brier term does not address at token level (inferred); 1/n normalisation (we use the mean) helps stability.
- Keep the query short: multi-vector lost badly on long queries (ArguAna). Row 210's CTX=1 query (transaction + 5 neighbours) moves toward that regime; CTX=0 vs 1 is an arm.
- Row 209: plain T5 trained into a strong multi-vector model needed MS MARCO scale; with our small data, start from retrieval- or ColBERT-pretrained checkpoints first (inferred).
