# Smarter, Better, Faster, Longer: A Modern Bidirectional Encoder for Fast, Memory Efficient, and Long Context Finetuning and Inference (ModernBERT)

- arXiv 2412.13663v2, 19 Dec 2024 - https://arxiv.org/abs/2412.13663
- Warner, Chaffin, Clavié, Weller, Hallström, Taghadouini, Gallagher, Biswas, Ladhak, Aarsen, Cooper, Adams, Howard, Poli (Answer.AI, LightOn, JHU, NVIDIA, HuggingFace)
- Venue: preprint
- Source: docs/papers/2412.13663/paper.txt

## One-paragraph summary
ModernBERT is an encoder-only family (base 149 M / 22 layers / hidden 768; large 395 M / 28 layers / hidden 1,024) that ports decoder-era design (RoPE, GeGLU, pre-norm, bias-free linears, alternating global/local attention with a 128-token window every non-third layer, full-model unpadding, Flash Attention, torch.compile) to MLM pretraining on 2 T tokens of English web, code and scientific text with a native 8,192-token context. It uses a modified OLMo BPE tokenizer (vocab 50,368, 83 unused tokens), 30 % masking, no NSP, StableAdamW and a warmup-stable-decay schedule. It is a Pareto improvement over BERT/RoBERTa on GLUE, BEIR (single- and multi-vector), long-document and code retrieval, and the fastest and most memory-efficient encoder at long context on an RTX 4090.

## Problem
Production encoders are still BERT-era: 512-token limit, no code, stale data, inefficient designs. MosaicBERT, CrammingBERT, NomicBERT and GTE-en-MLM each fixed only part of this.

## Method
- **Architecture** (Sec. 2.1, Table 4): bias only in the decoder linear; RoPE theta 160,000 global / 10,000 local; LayerNorm after embeddings; GeGLU; global attention every third layer; unpadding before the embedding layer; hardware-aware shapes chosen for a GPU basket that includes the RTX 3090 and 4090.
- **Data and tokenizer** (Sec. 2.2.1): mixture chosen by ablation; tokenizer keeps BERT special tokens (`[CLS]`, `[SEP]`, `[MASK]`) and templating; 83 unused tokens "to support downstream applications". The paper never states whether the tokenizer is cased; the OLMo BPE lineage is case-sensitive, but this must be verified on the tokenizer itself. Appendix D: the Llama 2 tokenizer degraded results; BERT/RoBERTa-era tokenizers were competitive on MNLI.
- **Training** (Sec. 2.2.2, Table 3): MLM 30 %; base LR 8e-4 for 1.7 T tokens at length 1,024; large initialised from base by centre tiling, LR 5e-4 then rolled back to 5e-5; context extension to 8,192 over 250 B tokens at 3e-4 plus a 50 B 1-sqrt decay on upsampled high-quality data; 194 h (base) / 425 h (large) on 8xH100; base final checkpoint is an average of the three best annealing checkpoints. Intermediate checkpoints released.
- **Evaluation** (Sec. 3): GLUE with per-task sweep (LR 1e-5 to 8e-5, WD 1e-6 to 1e-5, 1-10 epochs, early stopping; Table 6); BEIR DPR via sentence-transformers on MS MARCO hard negatives (1.25 M pairs, batch 16); ColBERT via PyLate with BGE-M3 distillation; MLDR; CodeSearchNet and StackQA.

## Experiments and results
- **Table 1**: base BEIR DPR 41.6 (GTE-en-MLM 41.4, BERT 38.9), ColBERT 51.3, MLDR-OOD ColBERT 80.2, GLUE 88.4 (first MLM model above DeBERTaV3-base 88.1), CSN 56.4, SQA 73.6. Large: DPR 44.0, ColBERT 52.4, GLUE 90.4 (DeBERTaV3-large 91.4), CSN 59.5, SQA 83.9. Single-vector MLDR-OOD trails GTE-en-MLM (27.4 vs 34.3 base), left open.
- **Table 2 (RTX 4090)**: base at 8,192 tokens: batch 98, 123.7 k tok/s fixed, 133.8 k variable vs GTE-en-MLM 38 / 46.8 k / 23.4 k; large 46.8 k vs 16.2 k. Base holds batches twice as large as any competitor.
- **Table 5 (GLUE, large)**: CoLA 71.4, SST-2 97.1, MRPC 91.7, STS-B 92.8, QQP 92.7, MNLI 90.8, QNLI 95.2, RTE 92.1.
- **Ablations (App. D)**: alternating attention equals full global attention at 100 B tokens; parallel attention hurt.

## Limitations
English only; MLM-only (RTD suggested); no parameter scaling beyond 395 M; long-context single-vector retrieval needs adapted tuning; not a generator, though the MLM head can fill a `[MASK]` (Sec. 6 cites Samuel 2024).

## Relevance to this workspace
- **MODEL-2.** The backbone for arm F: 8,192 context (whole field guide fits in the with-context condition), 83 reserved tokens (ModernBERT-Instruct uses `[unused0]` as its anchor; the rest are candidate single-token stand-ins for nonsense labels or entity names, initialised as unused rather than random), full fine-tuning of 395 M fits the RTX 3090 easily, and fine-tuning LRs of 1e-5 to 8e-5 for 1-10 epochs are documented. Note the 128-token local window: only every third layer can copy from a distant demo, which matters for in-context induction over a long field guide.
- **MODEL-3.** gte-modernbert-base inherits these properties and trains through the same sentence-transformers path as `exp_embed_vocab.py`.
- **MODEL-4 (casing).** The paper does not say "cased"; the tokenizer is OLMo-derived BPE, so it is expected to be case-sensitive. If verified, the 70.8 % bank-string transfer attributed to MiniLM's uncased WordPiece is not expected to survive, which makes ModernBERT the right cased control encoder.
- **MODEL-5.** ModernBERT's ColBERT (MaxSim) numbers are its strongest and the authors suggest a synergy between local attention and token-level matching; this supports testing token-max-sim retrieval for new-token identity instead of mean pooling.
- **REAL-2 / BASE-1.** A fast long-context encoder for indexing merchant records.

## Key references worth following up
2406.04823 (Samuel); 2402.00838 (OLMo); 2402.01613 (nomic-embed); 2304.13013 (StableAdamW); 2401.14489 (hardware-aware design); 2202.08005 (Wettig, should you mask 15 %); 2405.18392 (WSD scaling laws); 2410.02660 (long-context training); 2402.10171 (128k data engineering); 2107.02027 (sequence packing).
