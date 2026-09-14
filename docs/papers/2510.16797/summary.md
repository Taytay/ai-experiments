# MOSAIC: Masked Objective with Selective Adaptation for In-domain Contrastive Learning

- arXiv 2510.16797v2, 29 Jan 2026 (v1 Oct 2025) - https://arxiv.org/abs/2510.16797
- Vera Pavlova, Mohammed Makhlouf (burevestnik.ai, rttl.ai)
- Venue: preprint
- Source: docs/papers/2510.16797/paper.txt

## One-paragraph summary
MOSAIC adapts an already-trained general-domain embedding model (nomic-embed-text-v1-unsupervised) to a specialised domain in three stages: (1) add domain tokens to the tokenizer, initialised as the mean of their subword embeddings; (2) train jointly with a contrastive loss and an MLM loss whose softmax is restricted to the new domain tokens only, weighted alpha = 0.3, mask rate 0.15, with the masked inputs also fed to the contrastive branch; (3) finish with contrastive-only training to restore sentence-level discrimination. On the medical MTEB subset the pipeline moves the average from 53.94 to 54.64 (Table 1), and on a low-resource Islamic retrieval set from 32.05 (nomic) / 33.58 (contrastive-only adaptation) to 36.81 NDCG@10 (Table 3). Ablations show vocabulary expansion alone is harmful (47.76), all-token MLM is much worse than domain-restricted MLM at any alpha, and mask rate 0.3 or reversing stages 2 and 3 collapses performance (Table 2).

## Problem
Domain adaptation usually happens at the MLM pretraining stage and lacks domain vocabulary; adding tokens then continuing MLM destroys the contrastive properties of an embedding model, while adding tokens and training contrastively only gives the new rows "diluted signals due to the pooling functions" (Sec. 1). The MLM loss also dominates joint training because its softmax denominator (the full vocabulary) yields much larger gradients (Sec. 3.2).

## Method
- **Stage 1** (Sec. 3.1): train a domain tokenizer on the corpus, add the tokens missing from the base tokenizer (~9 k biomedical, 3 k Islamic), mean-of-subword init; encoder weights untouched.
- **Stage 2** (Sec. 3.2, Eq. 3-6): InfoNCE view of both objectives; restrict the MLM candidate set to V_D (domain tokens), with the output projection being the input embedding table e(x) (Eq. 5), so no separate head is needed; L = alpha * L_MLM_domain + L_CL; in-batch negatives; contrastive inputs carry the mask perturbation so the model must learn which tokens distinguish positives from negatives.
- **Stage 3** (Sec. 3.3): contrastive only, "corrective step".
- **Data**: ~20 M PubMed title-abstract pairs consistency-filtered with gte-base; Islamic: 7,587 verse pairs from Tafseer Ibn Kathir. Hyperparameters (Table 5): batch 128, max LR 5e-4, 1-5 epochs, one H100. Compute (Table 6): stage 2 about 2x a contrastive fine-tune; whole pipeline far below DAPT (10-15x).

## Experiments and results
- **Table 1 (biomedical MTEB average)**: nomic-unsup 53.940; naive in-domain contrastive continuation (nomic-embed-bio) 52.788; Stage1 47.756; Stage2 53.853; Stage3 54.638, best on BiorxivP2P, MedicalQA, MedrxivP2P, SciFact. Supervised MOSAIC-Bio-super has the highest supervised average; adapter-based E5-peft and contrastively trained PubMedBERT both underperform.
- **Table 2 (BIOSSES ablations on Stage 2)**: alpha 0.1 76.03, 0.2 81.34, 0.3 88.12, 0.4 86.79, 0.5 67.71; all-token MLM 63.99 (alpha 0.3) to 78.75 (alpha 0.001), all below contrastive-only 84.43; mask 0.3 gives 49.87; stages reversed 70.54.
- **Table 3 (Islamic, NDCG@10)**: Stage3 36.809 vs contrastive-only Islamic-embed 33.581, nomic 32.048, GTE 32.924, BGE 27.699; alpha 0.3 again best (Table 4). Fewer zero-score queries (Fig. 2).

## Limitations
Only one base model (BERT-style nomic-embed); domain tokens are frequent terms mined from large corpora, not rare entity names; evaluation is retrieval/STS/clustering, not entity recall; tokenizer casing is never stated; gains on the biomedical average are small and TRECCOVID regresses.

## Relevance to this workspace
- **MODEL-5 / BASE-5.** Our `ft_newtok_mean` arm is exactly MOSAIC Stage 1 + Stage 3 (mean-of-subword init, contrastive only), and MOSAIC's Table 1 reproduces our finding that this underperforms; their diagnosis is our diagnosis (pooling dilution). The missing piece is Stage 2: an MLM loss restricted to the 120 merchant tokens, weighted 0.3, run jointly with the contrastive loss, then a contrastive-only pass. Because Eq. 5 uses the input embedding table as the output projection, this works on all-MiniLM-L6-v2 without a saved MLM head.
- **MODEL-2 (encoder extension).** The reviewer's "MLM stage then contrastive" proposal should be run as MOSAIC's joint stage rather than sequential, since their stage-order ablation (88.1 to 70.5) shows joint-then-contrastive matters. Their masked-input trick also gives a free cloze probe: mask the merchant token and score over the 120-token merchant vocabulary, which is the reverse direction (`sells` / `reverse` tasks) of `Elrholm sells [MASK]`.
- **MODEL-4.** Domain tokens are added for both cased forms only if the tokenizer sees them; MOSAIC does not address casing, so the uppercase bank-string form still needs its own token or a normaliser (BASE-5, DATA-4).
- **MODEL-3.** The base for MOSAIC is BERT-style with 8 k context (nomic); gte-modernbert-base is the obvious modern substitute.
- **Caveat.** Their tokens each have thousands of occurrences; ours have 14-20 texts each, so the alpha and mask-rate optima may not transfer.

## Key references worth following up
2402.01613 (nomic-embed); 2212.03533 (E5); 2308.03281 (GTE); 2405.05374 (Arctic-embed); 2402.12036 (Belfathi, selective masking); 2007.15779 (PubMedBERT domain pretraining); 1807.03748 (InfoNCE); 1910.08350 (MI view of representation learning); SimCSE 2104.08821 (from memory).
