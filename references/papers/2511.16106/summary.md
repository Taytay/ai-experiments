# Incorporating Token Importance in Multi-Vector Retrieval (Weighted Chamfer)

- arXiv 2511.16106 (v1, 2025-11-20) - https://arxiv.org/abs/2511.16106
- Archish S, Ankit Garg, Kirankumar Shiragur, Neeraj Kayal (Microsoft Research India)
- Venue: AAAI-26 style file (venue inferred; none printed)
- Source: references/papers/2511.16106/paper_flat.tex (read 2026-10-05 for PLAN rows 209-210)

## One-paragraph summary
Weighted Chamfer replaces ColBERT's uniform average of per-query-token MaxSim with a weighted average, one scalar per vocabulary token (query side only), ColBERTv2 frozen. Zero-shot weights are BM25 IDF over the corpus; few-shot weights are learned with a softmax CE ranking loss, convex in the weights for fixed negatives (in practice hardest negatives re-mined each step, two negative-set sizes mixed). Reranking BM25's top 1,000 on 13 BEIR sets: average Recall@10 +1.28% relative with IDF, +3.66% few-shot (largest +14.27% on CLIMATE-FEVER); absolute gains mostly 0.5-4 points.

## Method
- Distance: sum_i w_{q_i} min_j ||q_i - d_j|| / n; uniform weights recover Chamfer.
- IDF(t) = log((N - n(t) + 0.5)/(n(t) + 0.5) + 1); unseen tokens 0; special tokens 0 or 1 chosen on validation.
- Few-shot: L = alpha CE(q; D+, Lambda1) + (1-alpha) CE(q; D+, Lambda2), Lambda1 within Lambda2; Adam lr 1e-4 cosine to 1e-8 over 100 iterations, uniform start, sum-to-1 projection; seen tokens keep their total IDF mass; learned weights kept only if they beat IDF on validation. Default |Lambda1| 10, |Lambda2| 100, alpha 0.1.
- Theory: recovery needs n >= Omega(log(T/delta)/lambda_min); VC bound with T+1.

## Experiments and results
- BM25 rerank top-10 (relative): Recall@10 +3.66%, MRR@10 +2.91%, nDCG@10 +3.01%. CLIMATE-FEVER 0.2804 -> 0.3204; SCIFACT 0.8136 -> 0.8652; NQ 0.7622 -> 0.7806; FEVER 0.9128 -> 0.9168.
- End-to-end ColBERTv2: Recall@10 +3.04%; several sets 0.00% (no weighting beat uniform on validation).
- Table inconsistencies noted (MS MARCO and DBPEDIA rows duplicated or mismatched). No seeds.

## Limitations
Weights by token ID only (context-free), query side only, encoder frozen; per-dataset searches; best-of-three selection inflates the reported gain.

## Relevance to this workspace
- In "payee | $12.47 | Wed" MaxSim weighs amount digits, separators and weekday like payee tokens. Token-ID weights fit poorly (digits are shared pieces); better: weights per field (payee, amount, weekday, separators), per-household IDF over payee strings (zero-shot, down-weights "POS", "DEBIT", "SQ *"), or a contextual weight head trained inside row 210 (li_decider.py QW=1: softmax over the query's tokens of a linear read of each token state).
- Expected size small (in-domain sets often 0); include as part of row 210 rather than its own row.
