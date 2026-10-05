# KaLM-Reranker-V1: Fast but Not Late Interaction for Compressed Document Reranking

- arXiv 2606.22807 (v3, 2026-09-22; v1 2026-06-22) - https://arxiv.org/abs/2606.22807
- Xinping Zhao, Jiaxin Xu, Ziqi Dai, Xin Zhang, Huiyao Chen, Shouzheng Huang, Xianhao Xiong, Danyu Tang, Xinshuo Hu, Guohong Fu, Meishan Zhang, Baotian Hu (HIT Shenzhen; Shenzhen Loop Area Institute; Soochow University)
- Venue: preprint (technical report, ICLR 2027 template)
- Source: references/papers/2606.22807/paper_flat.tex (read 2026-10-05 for PLAN rows 209-210). The paper never mentions "Jev" or decision models; the word appears only in its HF collection name ("lychee-kalm-reranker-and-jev").

## One-paragraph summary
"Fast but not late interaction" (FBNL): a T5Gemma2 encoder-decoder reranker whose encoder encodes each passage once offline, compressed by Matryoshka Embedding Pooling (mean of each r consecutive token states); the decoder reads instruction and query and attends to the cached passage states in one merged self+cross attention; relevance = softmax of the yes/no logits at the last position (pointwise). Three stages: SFT summed over r in {2, 8, 32, 128}; BCE distillation from Qwen3-Reranker-8B; a soup of two LoRA runs. BEIR nDCG@10 at r=4: Nano (0.27B active) 58.54 at 1.0x cost vs gte-reranker-base 56.77 at 8.0x; Small 61.07 vs Qwen3-Reranker-0.6B 59.36 (28.2x); Large 63.53 vs Qwen3-Reranker-4B 63.50 (157.8x).

## Method
- H_p = Enc(p); decoder Q = X W_Q, K,V = [X; H_p] W_{K,V}; only last-token logits kept.
- MEP: H^(r) = mean of groups of r tokens; loss summed over r with weight 1.
- Distillation: BCE to sigmoid(z_yes - z_no) of the teacher, summed over r.
- Data ~3.9M queries, 1 positive + >= 15 hard negatives; LoRA r 96 / alpha 48; group 16; lr 2e-4; batch 128 on 32 RTX 5090s; query 128 / passage 512 tokens.
- Cost model: 16.6x cheaper than a cross-encoder at n=256, 203.4x at n=4096 (analytic).

## Experiments and results
- BEIR: retriever 53.78; Nano 58.54; Small 61.07; Large 63.53 (Qwen3-Reranker-8B 65.11 at 359.8x). MIRACL: Nano 71.11, Small 74.06, Large 74.92 (Qwen3-8B 74.13).
- LMEB-Dialogue: Nano 61.07 over a 50.80 retriever; weak on temporal TMD (31.44 vs 58.79).
- Compression, Nano BEIR r = 2..128: 58.75, 58.54, 58.17, 57.47, 56.74, 55.88, 55.11. Without MEP training: 53.58 at r=2 falls to 38.57 at r=4; with it 57.72 at r=2.
- Cascade (top 100 at r=32, top 20 at r=2): Small 61.11 at 2.75x vs 61.17 at 8.65x.
- Stages (BEIR r=2): Nano 57.72 -> 58.51 (distil) -> 58.75 (soup); Small 59.57 -> 61.07 -> 61.17; MIRACL Nano +2.11 from distillation.

## Limitations
No MaxSim or ColBERT baseline, so "cross-attention beats late interaction" is argued, not measured; costs analytic; pointwise only, no calibration; base under Gemma terms (inferred: not an allowed licence here).

## Relevance to this workspace
- Row 210 must test cross-attention vs MaxSim itself on the same encoder; this paper gives no head-to-head.
- Caching saves little for our 15-60-token category documents; a per-option decoder pass costs 30-60 passes per transaction vs one MaxSim pass. A cheaper analogue: one or two cross-attention layers from transaction tokens to cached category token vectors, or one listwise pass over all of a household's category memories.
- If category memories are pooled, train with the pooling (MEP: -15 without).
- Distil soft labels from a stronger reader (+0.8 to +2.1): decider's option distribution as the teacher (check it is not saturated; see 2510.14880).
- Cheap screen, expensive rescore kept quality at a third of the cost: matches a kNN/MaxSim gate before decider.
- Not a candidate model: Gemma-licensed base.
