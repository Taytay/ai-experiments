# Mechanistic Fine-tuning for In-context Learning (ABFT)

- arXiv 2505.14233v2 (27 Sep 2025; v1 May 2025) - https://arxiv.org/abs/2505.14233
- Hakaze Cho, Peng Luo, Mariko Kato, Rin Kaenbyou, Naoya Inoue (JAIST, Beijing Institute of Technology, RIKEN)
- Venue: not stated in the text; ACL-style preprint
- Source: docs/papers/2505.14233/paper.txt

## One-paragraph summary
Attention Behavior Fine-Tuning (ABFT) improves few-shot classification ICL by placing the training loss directly on attention scores rather than on output logits. For each ICL prompt, the last-token attention row of every head is inspected; heads whose attention mass on the in-context label tokens exceeds a threshold are treated as induction heads and receive a loss that penalizes attention to wrong-label tokens and rewards attention to correct-label tokens. Only W_Q and W_K are updated (LoRA r=16 on 4-bit models above 10B), with 512 training prompts and 32 optimizer steps. Across 9 models from GPT2-Large (812M) to Llama3 56B and 8 datasets, ABFT raises average accuracy by roughly 10-20 relative points, beats MetaICL and PICL trained on 7,000x more data, is less harmful out of domain than end-to-end LoRA, and lands in the same loss basin as end-to-end and MetaICL models, suggesting end-to-end ICL training implicitly trains induction heads.

## Problem
Fine-tuning LMs on ICL-style data (MetaICL, symbol tuning) works but needs whole-model, full-precision training on large datasets. The authors want a low-resource alternative grounded in the induction-head account of ICL, where the prediction follows whichever label tokens the induction heads attend to.

## Method
Section 3. (1) Build k=4-shot prompts from a downstream dataset (nd=512). (2) Forward pass, collect attention matrices of all heads. (3) Head filter: with label positions I and last-row attention alpha, a head is an induction head if the summed attention on I exceeds T = k/(k + log n_t). (4) Loss on induction heads only: L(A) = A * sum over wrong-label positions of alpha_i + B * (1 - sum over correct-label positions of alpha_i), with A0=0.5, B0=1.0; A is adjusted online by a PID controller to keep the number of induction heads stable (Appendix A.3). (5) Backpropagate only into W_Q and W_K. Adam, lr 2e-5 (full W_Q/W_K) or 1e-4 (LoRA r=16 on 4-bit models), pseudo-batch 32, 32 steps. Label tokens are reduced to single tokens (Table 5). Everything runs on one A40 48GB.

## Experiments and results
- Table 1 (average over SST2, MR, FP, SST5, TREC, SUBJ, TEE, TEH): GPT2-L 45.86 to 63.21 (MetaICL 53.80, PICL 52.68); GPT2-XL 50.13 to 67.74; Falcon3 7B 67.55 to 79.59; Llama3 8B 66.60 to 80.20; DeepSeek-R1-Distill-Qwen 14B 72.61 to 78.21; Qwen2.5 32B 75.29 to 79.64; s1.1 32B 78.77 to 81.02; Llama3 42B 71.31 to 78.34; Llama3 56B 70.95 to 79.52. Contextual calibration is roughly neutral or negative.
- Table 2, ABFT vs end-to-end LoRA fine-tuning: Llama3 8B, E2E trains 0.5B parameters, 2.2x time, in-domain 78.33 / out-of-domain 61.74; ABFT 6.8M parameters, 1x, 72.54 / 64.34. DeepSeek 14B: E2E 78.26/63.62 vs ABFT 78.21/67.21. Qwen2.5 32B: E2E 82.09/62.24 vs ABFT 79.64/64.96. Llama3 56B: E2E 82.80/64.86 vs ABFT 79.52/67.32. E2E wins in domain by 0-6 points; ABFT harms out-of-domain less and costs about 1-3% of the trainable parameters and 40% of the time.
- Data efficiency (Figures 2, 15): at 512 samples or fewer ABFT and E2E are equal; E2E pulls ahead with more data.
- Consistency (Table 3): template consistency 86.93 to 90.32 and demonstration-sampling consistency 76.99 to 92.00 on Llama3 8B; similar on other models.
- Unseen-label setting (Figure 3, Appendix A.4): when all demonstrations carry a label different from the query's, ABFT still beats 0-shot although its induction heads are almost fully suppressed (Figure 10), so a non-copying channel exists.
- Ablations (Table 4, Falcon3 7B): full ABFT 79.59; no PID 76.86; no head filter 75.57; A=0 (no punishment) 72.13; B=0 (no reward) 56.47, i.e. below vanilla. Same ordering for Llama3 8B (80.20 full vs 58.79 with B=0).
- Loss-landscape analysis (Figures 7, 9): linear interpolation between pre-trained, E2E and ABFT parameters shows no high-loss barrier; MetaICL models lie in the same basin. Attention changes and W_Q/W_K shifts concentrate in middle layers (Figures 4, 8, 17-19).

## Limitations
Classification with a finite label set only; label words forced to single tokens; hyperparameters and the head-filter threshold are admitted to be unoptimized; each ABFT run is per dataset (in-domain), so it is an adaptation method, not a general ICL booster like symbol tuning; results are means of 2-4 repeats without CIs on Table 1; it depends on the induction-head account holding for the target model.

## Relevance to this workspace
- What carries ICL. Label induction in our episodes is, on this account, last-token attention from a small set of middle-layer induction heads to the label tokens of the demonstrations; the components to inspect or train are W_Q/W_K of those heads, and the loss can be defined on attention rows without touching the LM head. This gives a concrete diagnostic for TRAIN-3 (track induction-head count and correct-label attention per checkpoint) and for the arm A finding that knowledge text lowers the public ICL suite (does it disperse induction attention?).
- Cheaper training (TRAIN-1, TRAIN-4). The E stream could be replaced or supplemented by ABFT on 512 episodes for 32 steps, updating only W_Q/W_K; that removes the token-weight imbalance between K and E streams because the E objective no longer competes in the LM loss at all.
- MODEL-1. ABFT works at 812M and 1.6B, so induction-head training is not a scale-gated phenomenon in the way symbol tuning at 8B was; a 7B run is a reasonable next step.
- EVAL-4. The unseen-label protocol (demos deliberately carry only other labels) separates label copying from attribute reasoning; a Timmy variant where the query's group is not among the demo labels would test whether the model copies or induces.
- EVAL-5. Table 2's in-domain vs out-of-domain split (train on one dataset, test on the others) is a clean regression protocol and shows even attention-only tuning costs 2-4 OD points.
- BASE-4. ABFT on the 160 L3/L4 items would be a strong "task-specific adapter" comparison against the prototype classifier.

## Key references worth following up
2209.11895 (induction heads, Olsson et al.), 2501.15708 (STAICC evaluation library), 2110.15943 (MetaICL), 2303.03846, Cho et al. 2025a "Revisiting in-context learning inference circuit" (ICLR 2025; arXiv id likely 2410.04468, verify), Reddy 2024 (ICLR, mechanistic basis of ICL classification).
