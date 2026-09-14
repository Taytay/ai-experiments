# Large Language Models Are Not Robust Multiple Choice Selectors (PriDe)

- arXiv 2309.03882 (v4, 22 Feb 2024) - https://arxiv.org/abs/2309.03882
- Chujie Zheng, Hao Zhou, Fandong Meng, Jie Zhou, Minlie Huang (Tsinghua CoAI / WeChat AI)
- Venue: ICLR 2024
- Source: references/papers/2309.03882/paper.txt

## One-paragraph summary
LLMs answering letter-labelled MCQs have "selection bias": they prefer particular option IDs. Moving every gold answer to position D drops gpt-3.5-turbo on 0-shot MMLU from 67.2 to 60.9; moving it to A lifts llama-30B from 53.1 to 68.2 (Table 1). With 20 models on MMLU, ARC-Challenge and CSQA, the paper measures bias as the standard deviation of per-option recalls (RStd), shows via ablations that it comes mainly from token bias on the ID symbols rather than position, and proposes PriDe: estimate the model's prior over IDs from cyclic permutations on about 5% of test items, then divide it out of every remaining prediction (Eq. 3, 7, 8). PriDe cuts RStd by 5.6-7.6 points at 1.15x cost and raises accuracy 1.2-1.7 points on average (Table 3); the estimated prior transfers across domains (Figure 5).

## Problem
Symbol scoring (max probability over A/B/C/D) is the default in the harness, HF leaderboard and OpenAI Evals, but its output depends on which letter the gold sits under. Prompt fixes (debiasing instructions, CoT) do not help (Table 2). Dropping the IDs and scoring option contents (the cloze route) reduces bias but lowers accuracy, so a cheap, label-free correction for the standard format is needed.

## Method
- RStd: standard deviation of recalls over option positions; valid when gold positions are balanced (Section 2.2).
- Ablations (Section 2.4): shuffling the ID letters (so B can sit anywhere) barely changes RStd; removing IDs and scoring the option strings by length-normalised likelihood cuts RStd sharply (gpt-3.5-turbo MMLU 5.5 to 1.0; ARC 3.3 to 0.6, Table 2). Residual bias after ID removal is model-dependent; llama-2-13/70B even rise on MMLU and ARC.
- Decomposition (Eq. 2-3): P_observed(d_i | q, x_I) proportional to P_prior(d_i | q) times P_debiased(o_f(i) | q, x); the prior is assumed independent of option order, the debiased belief independent of the ID.
- Prior estimate (Eq. 7): softmax over i of the mean log P_observed(d_i | q, x_I) across cyclic permutations I, label-free. PriDe: compute this on K = alpha|D| items, average to a global prior, then P_debiased(o_i) approx P_observed(d_i) / P_prior(d_i) on the rest (Eq. 8, Algorithm 1).
- Baselines: Full Permutation (n! passes) and Cyclic Permutation (n passes), which average predictions over orderings.

## Experiments and results
- Prevalence (Section 2.3, Figures 3, 11-13): every model is biased; the direction varies across families and sizes, but each model's preference is similar across domains; 5-shot reduces but also alters it.
- Removing IDs (Table 3, 0-shot averages over 20 models): RStd -6.4 on MMLU with accuracy -2.1; ARC -5.1 / -2.9; CSQA -6.2 / -7.0. Section 2.5 concludes ID removal "is not a practical method to mitigate selection bias" because of the accuracy loss and inconvenience.
- PriDe at 5% (Table 3): MMLU RStd -7.6, accuracy +1.2 at 1.15x cost; Cyclic Perm -8.7 / +4.9 at 4x. ARC -5.6 / +1.3; CSQA -6.9 / +1.7. Priors are stable from 2% to 20% of items (Figure 17) and transfer across MMLU domains and ARC (Figure 5), with slight accuracy loss when the domain gap is large.
- How predictions change (Tables 5, 6): changed items had low original confidence and the debiased choice was usually top-2 originally; Cyclic Perm flattens distributions and flips more high-confidence predictions.
- falcon-7B and falcon-inst-7B show abnormally large bias and near-random MMLU, attributed to under-training (Figure 13 caption).

## Limitations
- Only 7B-70B open models plus gpt-3.5; nothing below 7B, and the smallest model studied (falcon-7B) is dismissed as under-trained.
- The prior is over option IDs, which this project does not use; the content-prior in cloze scoring is only touched by the "removing IDs" ablation.
- Assumes gold positions are balanced and that the debiased belief is order-invariant; 3.2% of MMLU items with "A and B"/"none of the above" were removed.
- PriDe is not designed to improve accuracy; gains are a side effect.
- Cloze baseline uses length-normalised likelihood without exploring normalisations.

## Relevance to this workspace
- EVAL-1: the project has no IDs, so token bias on letters does not apply; the paper's "removing IDs" condition is the project's scorer, and its result (bias falls but does not vanish, accuracy falls) is a warning that content-prior bias survives ID removal. It also argues against switching to symbol scoring, agreeing with the 2402.01781 summary.
- EVAL-2, how to detect option-prior bias: compute RStd over the fixed option vocabularies (8 types, 12 categories, 3 Timmy labels) from saved per-item predictions. This requires balanced gold labels, which the ladder has for types by construction; check category balance in the merchant set.
- How to debias: the PriDe decomposition applied to option contents gives a label-free content prior: average the per-item softmax over options across a 5-20% sample (or all 160 items), then divide each item's option probabilities by that average before argmax. This is contextual calibration with an empirical rather than content-free estimate and costs nothing extra. For Timmy items, where labels are bound to demo positions, cyclic permutation of the demo order (3 passes) is the exact PriDe recipe and cancels any "first demo's label" prior.
- EVAL-6: a base model that is above chance on unknowable held-out induction with RStd far from zero is exhibiting exactly the prior-driven behaviour this paper isolates; report RStd next to accuracy.
- STAT-2: RStd, prior estimation and Table 5-style change analysis all need per-item predictions saved.
- Below 7B and cloze: the paper says nothing about sub-7B models and treats cloze only as an ablation; the evidence that cloze reduces but does not remove bias is the transferable part.

## Key references worth following up
- Robinson and Wingate 2023, arXiv 2210.12353 (cloze vs symbol prompts).
- Pezeshkpour and Hruschka 2023, arXiv 2308.11483 (order sensitivity).
- Zhao et al. 2021, arXiv 2102.09690 (calibration with content-free inputs).
- Wang et al. 2023, Large language models are not fair evaluators, arXiv 2305.17926 (permutation averaging).
- Kadavath et al. 2022, arXiv 2207.05221 (calibration, cited for why the prior may misalign).
