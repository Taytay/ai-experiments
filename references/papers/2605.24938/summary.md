# Your Embedding Model is SMARTer Than You Think (SMART)

- arXiv 2605.24938 (v1, 2026-05-24) - https://arxiv.org/abs/2605.24938
- Jianrui Zhang (UW-Madison), Hyun Jung Lee (Korea University), Sukanta Ganguly (NetApp), Tae-Eui Kam, Donghyun Kim (Korea University), Yong Jae Lee (UW-Madison)
- Venue: preprint (NeurIPS 2026 style, preprint option)
- Source: references/papers/2605.24938/paper_flat.tex (read 2026-10-05 for PLAN rows 209-210)

## One-paragraph summary
Contrastive training on a single pooled vector also shapes the other tokens' hidden states (they lie on its gradient path), so a single-vector embedder's final-layer token states already support ColBERT-style MaxSim. SMART adds an L2-normalised final-layer MaxSim to the pooled cosine with unit weights (s_hybrid = s_single + s_late): no training, +0.30 to +2.54 average on MMEB-V2 retrieval across five backbones. A frozen-backbone LN+Linear token adapter adds about +1.1 more; one extra LoRA epoch with the hybrid objective gets within 0.63 of a from-scratch multi-vector model at ~80% of the training time; training with the hybrid objective from scratch beats pooled-only by +6.5 and late-only by +0.8. The authors find inference-only SMART "not beneficial" for classification-like tasks.

## Method
- Late score (Eq. 4): mean over query tokens of the best cosine against candidate tokens; final-layer states, L2-normalised, padding and pooling token excluded on both sides. Hybrid (Eq. 5): unit weights.
- Adapter: r_i = normalize(Linear(LN(h_i))), backbone frozen, trained with the late score only, scored hybrid; ColPali training set, batch 512 (output dimension not given).
- LoRA conversion: LamRA recipe on Qwen3-VL-2B (r 128, alpha 256, lr 1e-4, batch 512); Convert = Single + 1 epoch hybrid.

## Experiments and results
- Toy local-binding test: pooled 31.9%, late 56.8%, hybrid 42.6%, jina-v4 50.9%, ColPali 48.7%.
- MMEB-V2 averages before -> after: VLM2Vec-V2.0 64.50 -> 67.04; GME-2B 69.00 -> 70.00; GME-7B 72.26 -> 72.56; Qwen3-VL-Embed-2B 74.87 -> 75.77; -8B 78.83 -> 79.34.
- Adapter (visdoc): Qwen3-VL-Embed-2B 79.27 -> 80.10 (SMART) -> 81.25 (adapter); 8B 82.33 -> 82.88 -> 83.89.
- LoRA (visdoc): Single 72.60; + SMART 74.18; Convert 77.68 (9.5 h); Multi 78.31 (12 h); Hybrid from scratch 79.10.
- Layer (Qwen3-VL-Embed-2B, pooled fixed at layer 28): token layer 20 best (80.16), 28 gives 80.10, base 79.27: any layer from ~70% depth works.
- No seeds or variance.

## Limitations
Not beneficial inference-only for classification or low-entropy targets (forced token matching "can actively introduce noise"); multimodal 2-8B backbones and long inputs only; unit weights never tuned; adapter dimension and schedule missing.

## Relevance to this workspace
- A free test on what we have: our history encoders (bge-small, CLS pooling, InfoNCE on CLS) are exactly SMART's setting. Re-score kNN neighbours with CLS cosine + final-layer token MaxSim (layer 12; also try 9-10), read with hist_fast. Expect small or null: inputs are ~10 tokens and the end task is classification, where the authors saw no gain; read paired.
- Row 210: the hybrid objective (pooled cosine + MaxSim) is the strongest result (+6.5 over pooled-only, +0.8 over late-only); li_decider.py HYBRID=1 adds the normalised [CLS] cosine to MaxSim. Starting from an already-trained encoder and fine-tuning briefly (Convert) is close to training from scratch; row 210 starts from row 195's ColBERT.
- The unit weighting assumes both scores share one cosine space: true for CLS + final-layer states, not for a projected 128-d reader, where the mix may need a weight.
