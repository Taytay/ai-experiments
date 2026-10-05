# Fantastic (small) Retrievers and How to Train Them: mxbai-edge-colbert-v0 Tech Report

- arXiv 2510.14880 (v1, 2025-10-16) - https://arxiv.org/abs/2510.14880
- Rikiya Takehi (Mixedbread AI; Waseda), Benjamin Clavié, Sean Lee, Aamir Shakir (Mixedbread AI)
- Venue: preprint (LNCS template)
- Source: references/papers/2510.14880/paper_flat.tex (read 2026-10-05 for PLAN rows 209-210)

## One-paragraph summary
Two small ColBERTs, 17M (projection 48) and 32M (64), on Ettin encoders (ModernBERT recipe). A dense embedder is built first (contrastive pre-training on 197M pairs; AnglE fine-tuning on mined hard negatives; L2 distillation to StellaV5-1.5B's vectors), then the ColBERT stage is trained with PyLate (KL to teacher scores, 16-way tuples, MS MARCO only) with ablations. NanoBEIR: 17M 0.6405, 32M 0.6520, ColBERTv2 0.6198, answerai-colbert-small-v1 (33M, dim 96) 0.6545. BEIR: 32M 0.521, answerai-colbert-small 0.534, bge-small-en-v1.5 0.517, GTE-ModernColBERT 0.547. The win is long context (LongEmbed 0.849 vs 0.441 answerai, 0.312 bge-small).

## Method
- Dense warm start (not ablated here; cited prior work): GradCache, batches 24,576 / 12,288; AnglE fine-tuning with hard negatives mined by Qwen3-Embedding-8B (threshold 0.95) mixed with 35% BM25 and 30% random; L2 distillation through a discarded 2-layer projection.
- ColBERT stage: PyLate, 16-way tuples, batch 128, KL on normalised teacher scores, documents 220 tokens.

## Experiments and results
- Ettin needs a higher lr: 17M 3.5e-4 -> 0.493, 6e-4 -> 0.523; 32M 2.8e-4 -> 0.543, 5e-4 -> 0.559.
- Dense stages: fine-tuning 17M 0.523 -> 0.556, 32M 0.559 -> 0.576; distillation 17M -> 0.567, 32M -> 0.626.
- Teacher: BGE-Gemma2 reranker 0.6286; Qwen3-Reranker-8B 0.5991 (scores saturated at 0/1; temperature did not fix it).
- Muon 0.5985 vs best AdamW 0.5923 (lr-sensitive). Distilled base helps ColBERT: 0.5771 -> 0.5911.
- Projection dim (32M): 96 0.5991, 64 0.5985, 48 0.5967, 32 0.5772, 24 0.5423, 16 0.5126.
- Projection head (17M): 2-layer FFN with upscaled hidden and residual 0.6405 vs linear 0.6275 (+1.3, stable across seeds).
- Lowercasing: 17M +0.9 (consistent across seeds); 32M no effect.
- Efficiency (NanoBEIR, RTX 4090): 17M 51 s, 275 MB per 10k docs; answerai-small 59 s, 549 MB.

## Limitations
Mostly single-run ablations on NanoBEIR subsets; ColBERT stage on MS MARCO only; the released models do not beat answerai-colbert-small-v1 on short text; licences not stated (Apache-2.0 for Mixedbread, Answer.AI, LightOn and MIT for Ettin, bge-small, inferred; check with open_licence).

## Relevance to this workspace
- Row 209: on short text the incumbents hold (answerai-colbert-small-v1 best small ColBERT; bge-small level with mxbai-edge-32m), so expect small gains from Ettin; answerai-colbert-small and GTE-ModernColBERT are the likelier arms. With Ettin, raise lr ~1.7-1.8x. Ettin and ModernBERT are cased: lowercase bank strings (often ALL CAPS) for small cased bases, or run both.
- Projection: 384 -> 128 can drop to 64 or 48 at no measurable cost (not 32); a 2-layer residual FFN head +1.3 over linear.
- Warm start the token reader from a trained dense encoder (row 210 starts from row 195's ColBERT, itself bge-small).
- Distillation teacher must not be saturated: check how peaked decider's softmax is before distilling it (row 210), or keep a hard-label term.
