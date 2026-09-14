# AlphaEdit: Null-Space Constrained Knowledge Editing for Language Models

- arXiv 2410.02355v4 (22 Apr 2025) - https://arxiv.org/abs/2410.02355
- Junfeng Fang, Houcheng Jiang, Kun Wang, Yunshan Ma, Jie Shi, Xiang Wang, Xiangnan He, Tat-Seng Chua (USTC, NUS)
- Venue: ICLR 2025
- Source: docs/papers/2410.02355/paper.txt

## One-paragraph summary
AlphaEdit keeps the MEMIT machinery but projects each weight perturbation onto the null space of the covariance K0 K0^T of preserved-knowledge keys before applying it, so that (W + Delta P) K0 = W K0 exactly (Eqn 7, 10). This removes the preservation term from the objective and lets the solver focus on the new facts (Eqn 11-14). In sequential editing (2,000 edits in batches of 100) on GPT2-XL, GPT-J and LLaMA3-8B, it lifts efficacy and generalization by 12.5 and 16.8 points on average over the best baseline (32.9 / 30.6 on LLaMA3), keeps fluency near the pre-edit value, and, unlike MEMIT, RECT and PRUNE, preserves GLUE-style general capability through 3,000 edits (Table 1, Figure 4).

## Problem
Locate-then-edit methods balance update error e1 against preservation error e0 with a weighted sum; in practice e1 dominates, edited models overfit, hidden representations drift (Figure 1b), and under repeated edits the model forgets and collapses (Figure 1c, Gupta et al. 2024).

## Method
- Compute K0 from 100,000 Wikipedia triplets (as in MEMIT), then SVD of K0 K0^T (d0 x d0, intermediate MLP width) and keep eigenvectors with eigenvalue below 1e-2 to form P = U U^T (Eqn 8-9, footnote 1).
- Objective: min ||(W + Delta P) K1 - V1||^2 + ||Delta P||^2 + ||Delta P Kp||^2 where Kp are keys of previously edited facts (Eqn 12); closed form Delta_AlphaEdit = R K1^T P (Kp Kp^T P + K1 K1^T P + I)^-1 (Eqn 14), versus MEMIT's R K1^T (Kp Kp^T + K1 K1^T + K0 K0^T)^-1 (Eqn 15). One extra matrix product; P is computed once per layer.
- Layers: GPT2-XL [13-17], GPT-J [3-8], LLaMA3-8B [4-8]; lambda 15,000-20,000; 20-25 steps at lr 0.5 (0.1 for LLaMA3); single A40 48 GB (Appendix A.3).

## Experiments and results
- Table 1 (2,000 sequential edits, batch 100). LLaMA3-8B CounterFact: MEMIT Eff 65.65 / Gen 64.65 / Spe 51.56 / Flu 437.43; RECT 66.05 / 63.62 / 61.41 / 526.62; AlphaEdit 98.90 / 94.22 / 67.88 / 622.49 (pre-edit Spe 89.48, Flu 635.23). GPT-J: MEMIT 98.55 / 95.50 / 63.64 / 546.28 vs AlphaEdit 99.75 / 96.38 / 75.48 / 618.50. GPT2-XL (1.5B): MEMIT 94.70 / 85.82 / 60.50 / 477.26 vs AlphaEdit 99.50 / 93.95 / 66.39 / 597.88. zsRE LLaMA3: MEMIT Eff 34.62 vs AlphaEdit 94.47.
- General capability (Figure 4, six tasks SST, MRPC, CoLA, RTE, MMLU, NLI): all baselines approach zero F1 after 2,000 edits on LLaMA3; AlphaEdit stays at the original level to 3,000 edits.
- Hidden-state t-SNE (Figure 5, Appendix C.3): AlphaEdit shows minimal distribution shift; RECT on LLaMA3 reverses the distribution.
- Plug-in (Figure 6-7): adding the projection to MEMIT, PRUNE and RECT gives +28.24% editing and +42.65% general-capability on average.
- Small models (Table 3, Appendix C.6): on Gemma and phi-1.5 MEMIT Eff 64.68 / 55.71 and fluency 373.94 / 368.57; AlphaEdit 75.21 / 70.79, fluency 398.96 / 399.47. Smaller models degrade far more than 6-8B ones under 2,000 sequential edits.
- KnowEdit wiki_recent on LLaMA3-8B-Instruct (Table 4, with norm-outlier filtering): MEMIT Edit Succ 56.25 / Portability 42.73 / Locality 41.02; AlphaEdit 96.10 / 57.30 / 54.76. MQuAKE multi-hop (Table 5): MEMIT 3.35, AlphaEdit 5.03 (CoT 9.14) on GPT-J, i.e. multi-hop use of edited facts is near zero for everyone.
- K0 sample size (Table 6): efficacy and generalization stable down to 10% of the data; specificity falls 11.76 points on LLaMA3.
- Runtime (Table 7): per 100 edits LLaMA3 222.51 s (MEMIT) vs 223.24 s (AlphaEdit); GPT-J 334.74 vs 334.93; GPT2-XL 474.14 vs 476.79.
- Memory-based baselines (Table 2): GRACE reaches Eff 96.72 but Gen 50.14 on LLaMA3; AlphaEdit wins Eff/Gen, not always Spe/Flu.

## Limitations
Only the sequential batched regime is reported; single-batch insertion of a few hundred facts is not separately characterised. Specificity still ends 15-22 points below pre-edit. Portability (wiki_recent 57.30) and multi-hop (single digits) remain poor. Not tested on multimodal or reasoning models (Section 6), nor on Qwen. As 2511.05852 later shows, the null-space storage is the most fragile to subsequent fine-tuning.

## Relevance to this workspace
Informs BASE-2, EVAL-5, TRAIN-3, TRAIN-4, MODEL-1.
- New subjects: wiki_recent (post-2022 facts) is the closest test; edit success 96.10 but portability 57.30 and locality 54.76.
- Scale and size: for a 3B-class model, the Gemma / phi-1.5 rows are the relevant data: 2,000 sequential edits already cut fluency by roughly a third even with AlphaEdit; 680 edits in one or a few batches should be far gentler, but this must be measured.
- Locality: AlphaEdit's general-capability curves (F1 versus number of edits on SST, MMLU, NLI) are the template for EVAL-5's out-of-distribution locality and for TRAIN-3-style curves; our ICL suite and an MMLU slice are the direct analogues. Generality = paraphrase = bare-format L1 recall. Portability = L2-L4 manipulation and reverse tasks, where the paper's own numbers are weak.
- Survival under LoRA: not studied here; 2511.05852 finds AlphaEdit edits decay more than MEMIT's.
- Qwen: not tested. Cost: identical to MEMIT (Table 7); the projection needs one SVD of a d0 x d0 matrix per edited layer (d0 = 11008 for Qwen2.5-3B), which is feasible on 24 GB.
- Practical verdict: run AlphaEdit as the second editing arm, since it is a one-line change in EasyEdit and small models are where MEMIT degrades most.

## Key references worth following up
2401.07453 (editing at scale causes forgetting), 2401.04700 (RECT: editing harms general abilities), 2405.16821 (PRUNE), 2403.07175 (Rebuilding ROME), 2502.05628 (AnyEdit), 2410.04045 (neuron-level sequential editing), 2305.12740 (IKE, in-context editing), 2409.05806 (CKnowEdit).
