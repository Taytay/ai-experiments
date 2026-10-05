# Unified Work Embeddings: Contrastive Learning of a Bidirectional Multi-task Ranker (UWE)

- arXiv 2511.07969 (v2, 2026-04-07; v1 2025-11-11) - https://arxiv.org/abs/2511.07969
- Matthias De Lange, Jens-Joris Decorte, Jeroen Van Hautte (TechWolf)
- Venue: preprint (ACL style, 9 pages)
- Source: references/papers/2511.07969/paper_flat.tex (read 2026-10-05 for PLAN rows 209-210)

## One-paragraph summary
WorkBench frames six labour-market tasks as ranking over label spaces up to ~14k ESCO skills; UWE is one bi-encoder (all-mpnet-base-v2, 109M) with a symmetric, multi-positive "many-to-many" InfoNCE split per bipartite graph (skill-job, skill-vacancy sentence, skill-alias) and a **soft late interaction** scorer: a temperature softmax over token-to-token similarities replaces hard MaxSim. Task-average MAP 40.2, +2.6 over task-specific ContextMatch and +4.4 over Qwen3-8B embeddings with 73x fewer parameters; zero-shot over unseen ontologies (O*NET 35.1 vs 26.2/25.7). In its own ablation hard MaxSim scored *below* mean pooling (37.63 vs 38.00) while two-sided soft interaction scored 39.45.

## Method
- Loss: L_{Y|Q} = expectation over q and over each in-batch positive y+ of -log softmax (SupCon L_out); symmetric L_{Y,Q} = L_{Y|Q} + L_{Q|Y}; total alpha_J L_{S,J} + alpha_V L_{S,V} + alpha_A L_{S,A} with alpha (1, 0.5, 0.5); tau 0.05 (skill-job), 0.02 (others).
- In-batch negatives only; batch 512 skills + 512 each of J, V, A; vacancy sentences padded with a random non-matching sentence with p 0.8.
- Soft late interaction: A = rowwise softmax(E_q E_y^T / tau_a) on raw dot products; sim = <A, cosine matrix>_F (sum over query tokens); tau_a -> 0 gives MaxSim. Best tau_a 0.1 per the ablation text (the appendix bolds 0.5: inconsistent).
- AdamW, warm-up 10%, peak lr 8e-5 (appendix bolds 8e-4: inconsistent), max 64 tokens, one A100-40GB; 3.3M job-skill pairs after GPT-4o-mini enrichment of rare skills.

## Experiments and results
- MAP / RP@10: UWE 40.2 / 57.7; ContextMatch 37.6 / 53.7; MPNet base 32.3; EmbeddingGemma 34.4; Qwen3-0.6B 32.1, -4B 34.8, -8B 35.8. Fine-tuning MPNet +7.9; scaling Qwen3 0.6B -> 8B +3.7.
- Latency: 15.9 ms/query vs 13.2 plain MPNet and 60.2 Qwen3-8B; soft late interaction ~1.9x FLOPs.
- Loss ablation (5 seeds): plain InfoNCE 36.40 +- 0.07; symmetric + structure 38.17 +- 0.18; full 38.33 +- 0.24 (multi-positive alone inside the CI).
- Scorer ablation (5 seeds): mean-pooled 38.00 +- 0.41; MaxSim 37.63 +- 0.52; soft, target mean-pooled 39.12 +- 0.10; soft both sides 39.45 +- 0.20.
- No synthetic data -1.7; zero-shot O*NET +8.9 MAP, SkillsFuture +2.1.

## Limitations
English; synthetic and self-tagged data; checkpoint picked on WorkBench validation; two hyperparameter inconsistencies; no hard negatives; no calibration.

## Relevance to this workspace
- Soft late interaction over hard MaxSim, on short inputs (<= 64 tokens): copy into row 210 (li_decider.py SOFT=tau_a: A = softmax over document tokens of raw dot / tau_a, score = sum of A x cosine, averaged over query tokens) and grid tau_a {0.05, 0.1, 0.2}. Our ColBERT reader (69.5%) no better than vector readers fits their MaxSim < pooled finding.
- The same softening one level up: a temperature-softmax over a category's filings instead of the hand-picked "mean of the best 3".
- Symmetric loss (+1.77 from symmetry and structure) for the history encoders; same-category filings in a batch as positives, never negatives (the dedup in row 202).
- Text-described labels ranked zero-shot in unseen label spaces support scoring per-household categories by their text plus filings.
- Rare-label enrichment helped (-1.7 without); judge any analogue on blind sets.
- Row 209: the training objective mattered more than the base (+7.9 fine-tuning vs +3.7 scaling 13x).
