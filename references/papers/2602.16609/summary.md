# ColBERT-Zero: To Pre-train Or Not To Pre-train ColBERT models

- arXiv 2602.16609 (January 2026) - https://arxiv.org/abs/2602.16609 ; blog https://huggingface.co/blog/lightonai/colbert-zero
- LightOn (PyLate authors); models, every intermediate checkpoint (with and without prompts) and training scripts released, Apache-2.0
- Source: references/papers/2602.16609/paper.txt (pypdf extraction, read 2026-10-06 for PLAN rows 222 and 209; owner: "This is definitely an important read")

## One-paragraph summary
Late-interaction (ColBERT) models are usually made by a short knowledge-distillation (KD) step on top of a dense model that had all its
contrastive training in the single-vector setting. With ModernBERT-base and Nomic Embed's public data, the paper runs the three usual phases
(unsupervised contrastive with in-batch negatives at batch 16k; supervised contrastive with mined hard negatives; KD from a reranker, KL
loss) in the multi-vector setting instead. Full multi-vector pre-training (ColBERT-Zero) reaches 55.43 nDCG@10 on BEIR, above
GTE-ModernColBERT (54.67) and its stronger-data dense base gte-modernbert (55.33); KD alone on the dense supervised model reaches 54.09
(-1.3). Doing only the supervised contrastive phase plus KD in the multi-vector setting, from the dense *unsupervised* model, reaches 55.12:
99.4% of the full result for a tenth of the compute (40 vs 408 GH200-hours). Prompts matter: Nomic's models were pre-trained with
"search_query: " / "search_document: " prefixes; fine-tuning without them, or adding them to a model pre-trained without, costs
noticeably; the authors conjecture the prompt tokens act as implicit query expansion (global "placeholder" tokens), which Flash-Attention
models otherwise lose because masked-token embeddings are no longer usable.

## Numbers
- Table 1 (hyperparameters): unsupervised LR swept 3e-3..1e-5, batch 16,384, temperature 0.2 (learnable, then fixed), query 39 / document
  187 tokens, 368 GPU-hours; supervised LR 2e-5..8e-8, batch 64, temperature 0.2, document 519; KD LR 1e-3..1e-7, batch 128 (accumulated 2).
- Table 2 (BEIR avg nDCG@10): ModernBERT-embed unsupervised 47.05, supervised 52.89; gte-modernbert 55.33; GTE-ModernColBERT 54.67; KD on
  dense supervised 54.09; supervised + KD in ColBERT 55.12; full ColBERT pre-training + KD 55.43.
- Prompts: both 7 tokens; lengths raised by 7 to fit them; prompt-free ColBERT-Zero "significant drop"; with stronger supervised data
  (NV-Retriever) the misalignment mattered less.

## What it means for us (PLAN rows 209, 212, 221, 222)
- **Do the contrastive stages in the late-interaction setting, not pooled.** Row 222's knowledge and alias stage (knowledge_stage.py) trains
  pooled sentence vectors (InfoNCE on mean pooling) and only the history stage uses MaxSim: the setup the paper finds 1.3 points short. A
  MaxSim version of the knowledge/alias stage is the direct test (arm for row 222).
- **The cheap path is enough:** supervised contrastive (our decision training with hard negatives = the household's other categories) then
  KD. Our missing piece is the KD step: distilling decider's (or the 35B's) option distribution into the late-interaction model (row 212 (d),
  row 218's "distil decider" idea, LITE 2406.17968).
- **Keep the base's markers.** mxbai-edge-colbert-v0 (Ettin-based, Apache-2.0) uses "[Q] " / "[D] " prefixes (config_sentence_transformers.json);
  li_decider now has PREFIX=1. Our own chain (hist_colbert_v1 -> a1 -> ...) never used prefixes, so it stays without.
- Prefix tokens as global "memory" slots may be why short queries plus per-filing candidates (a5) beat long history queries on the owner's
  budget: worth an arm with a few learned global tokens (row 211's memory tokens were the related idea).
