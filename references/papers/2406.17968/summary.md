# Efficient Document Ranking with Learnable Late Interactions (LITE)

- arXiv 2406.17968 (v1, 2024-06-25) - https://arxiv.org/abs/2406.17968
- Ziwei Ji, Himanshu Jain, Andreas Veit, Sashank J. Reddi, Sadeep Jayasumana, Ankit Singh Rawat, Aditya Krishna Menon, Felix Yu, Sanjiv Kumar (Google)
- Venue: preprint
- Source: references/papers/2406.17968/paper_flat.tex (read 2026-10-05 for PLAN rows 209-210)

## One-paragraph summary
LITE replaces ColBERT's sum-of-max reduction of the token similarity matrix S = Q^T D with a learned MLP: "separable LITE" applies a shared MLP to each row, then to each column, then a linear map to a score. It is a universal approximator even with 2 pooled tokens per side, while a dot-product dual encoder of dimension < P.L has error >= 1/20 on some score. With a 6-layer 768-d BERT distilled from a cross-encoder: MS MARCO MRR@10 DE .355, ColBERT .383, LITE .393, CE student .395; LITE beats ColBERT on 11 of 14 BEIR sets zero-shot; storing 50 instead of 200 document tokens keeps .391 at 56 ms vs ColBERT's 62 ms and 0.25x storage.

## Method
- Row step: S'_{i,:} = LN(ReLU(W2 LN(ReLU(W1 S_{i,:} + b1)) + b2)), W1 (m2, L2), shared across rows; column step likewise with W3 (m1, L1); score = w^T vec(S''). m1 360, m2 2400. Fixed L1 30, L2 200; position-sensitive.
- Losses: KL to the teacher's softmax over (positive, negatives); margin-MSE; one-hot CE (NQ only). AdamW, batch 128, lr 2.8e-5, 1.5M steps.

## Experiments and results
- MRR@10 MS MARCO / DL19 / DL20 / NQ: DE .355 / .861 / .842 / .699; ColBERT .383 / .878 / .860 / .756; LITE .393 / .898 / .873 / .769.
- Loss: NQ one-hot / KL / margin-MSE: ColBERT .690 / .754 / .756; LITE .710 / .741 / .769. Without a teacher LITE beats ColBERT by +2.0; distillation is worth far more than the scorer (+6.6 for ColBERT).
- BEIR: LITE > ColBERT on 11/14 (Quora .839 vs .767); loses HotpotQA, ArguAna, SciFact.
- ColBERT top-k aligned tokens k = 1, 2, 4, 8: .383, .378, .380, .382 (no gain). Frozen encoders: ColBERT .112, LITE scorer-only .188. KNRM .390.

## Limitations
No hard-negative mining; reranking only; fixed lengths, position-sensitive, slower than sum-of-max at equal storage (111 vs 62 ms); existence-only theory; one encoder, no seeds.

## Relevance to this workspace
- Distil decider into row 210: the teacher mattered more than the scorer (.690 -> .756). decider's per-option log-probs exist for the test sets (results/per_item); training households would need a decider read first. A KL or margin-MSE term beside CE and Brier.
- A learned scorer beats hand-written MaxSim even with a frozen encoder (.112 -> .188): a cheap arm is a small separable-LITE head over S = transaction tokens x category-document tokens (widths far below 360/2400), or a "LITE over filings": the sorted top-K filing cosines per category through a tiny MLP, generalising our "mean of the best 3".
- Position sensitivity: order a category document's filings canonically (by recency, as li_decider.py does, or by similarity).
- Top-k token MaxSim does not help (k 2-8 no better than 1).
