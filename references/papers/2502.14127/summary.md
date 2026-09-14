# Which of These Best Describes Multiple Choice Evaluation with LLMs? A) Forced B) Flawed C) Fixable D) All of the Above

- arXiv 2502.14127 (2025) - https://arxiv.org/abs/2502.14127
- Nishant Balepur, Rachel Rudinger, Jordan Boyd-Graber (University of Maryland)
- Venue: ACL 2025 position paper (the extracted text prints no venue line)
- Source: references/papers/2502.14127/paper.txt

## One-paragraph summary
A position paper, built on a PRISMA-style review of 1,250 papers filtered to 122 (Appendix A.1), arguing that MCQA is over-used and under-designed for LLM evaluation. The format cannot test generation or subjectivity, mismatches real usage (MCQA is 32% of HELM, 71% of the GPT-4 card and 79% of the Open LLM Leaderboard, while over 90% of ShareGPT/WildChat queries are generative), and rewards recall over deeper knowledge (Section 3). The authors propose two generative variants that keep cheap scoring: Constructed Response (drop the choices, score a short answer against the gold) and Explanation MCQA (pick and justify). They then list dataset failures (leakage, unanswerable items, shortcuts, saturation) with fixes from educational testing (item-writing rubrics, calibrated scoring, contrast sets, choices-only baselines, Item Response Theory), and connect LLM robustness, bias and unfaithful-explanation failures back to those flaws (Sections 5-6).

## Problem
MCQA is chosen for scoring convenience, not validity. Choices leak answers (validation is easier than generation and the two are inconsistent in LLMs), distractors can be cheated with partial inputs, saturated sets stop discriminating, and letter-based prompting adds symbol and position artifacts that measure binding rather than knowledge (Section 6.1: "symbol binding error: LLMs 'know' the answer but cannot link it to the right choice").

## Method
Argument and synthesis rather than experiments. Key devices: (1) Constructed Response conversion of existing MCQs by omitting choices and matching the generated short answer to the gold, citing ARC-DA and Myrzakhan et al. 2024 for the two hurdles (which MCQs convert, how to score); (2) Explanation MCQA scored like reasoning tasks; (3) Haladyna and Downing's item-writing rubric (e.g. rule 37 "one and only one correct option", Figure 4) applied by an LLM to flag unanswerable MMLU items; (4) calibrated scoring (probability scoring, negative marking, elimination scoring) to deter guessing; (5) shortcut detection via choices-only accuracy, uniform data design, contrast sets (Figure 5) and adversarial "cheating" models; (6) IRT to separate flawed items (negative discriminability) from genuinely hard ones and MIRT to name latent skills.

## Experiments and results
- No new benchmark experiments; the figures are illustrative GPT-4o/o1 prompts (Appendix A.2).
- Quantitative claims are cited: users rate a distractor as most plausible in over 20% of commonsense MCQs (Palta et al. 2024); GPT-3 saw 45% of RACE's test set (Sainz et al. 2023); HellaSwag has the highest known choices-only accuracy because answers and distractors come from different generators (Section 5.3.2).
- Section 6.1 notes that probability-based scoring "seems more at fault, more sensitive to prompts" than generation for instruction-tuned models (Wang et al. 2024a) and that logically equivalent probability and generation evaluations give different answers (Lyu et al. 2024).
- Section 6.2: selection biases toward symbols, positions and phrases like "none of the above" are attributed to shortcuts learned from training data.

## Limitations
- No experiments of its own; every number is second-hand.
- Recommendations (rubrics, IRT, CR scoring) presume many models or many items; with one model family and 160 items per level, IRT is underpowered.
- Free-generation scoring needs an answer-equivalence judge; the paper points to PEDANTS and LLM judges but does not solve it.
- Aimed at instruction-tuned chat models and public benchmarks; base-model cloze scoring is mentioned only as the pre-instruction-tuning practice.
- Nothing on normalisation or option priors beyond citing Zheng et al. and Alzahrani et al.

## Relevance to this workspace
- EVAL-3: the paper's Constructed Response proposal is the missing generative check. Concretely: for L1, L3 and the merchant category items, greedy-decode after "Answer:", normalise, match to the gold string (exact, then fuzzy), and report generator-validator agreement with the cloze prediction. The project's production task is generative, which is the paper's central argument for why MCQA alone under-tests it.
- EVAL-4: "one and only one correct option" is rule 37; Timmy items with one demo per group violate it whenever demos differ on more than one attribute. The fix in QUESTIONS.md (two demos per group sharing exactly one attribute) is the rubric applied.
- EVAL-6: a base model at 42.7% on an unknowable task is the paper's "choices-only cheater" signature. The cloze analogue of choices-only accuracy is scoring the options with the question content removed (Holtzman's UNC). Items where that beats chance should be flagged; IRT's negative-discriminability filter is the multi-model version.
- EVAL-7: distractors from other categories are the "not shortcut-proof" failure; same-category distractors are the rubric's fix, and a contrast set (same options, different question, different gold) is the check.
- EVAL-1/EVAL-2: the paper endorses calibration scoring beyond accuracy; the project's saved margin (`accuracy()` lines 131-132) is a start. Symbol scoring is discouraged for knowledge testing (symbol binding), consistent with keeping cloze.
- Below 7B and cloze: not addressed; the arguments transfer only insofar as small base models are asked to validate rather than generate.

## Key references worth following up
- Myrzakhan, Bsharat, Shen 2024, Open-LLM-Leaderboard: from multi-choice to open-style, arXiv 2406.07545.
- Bhakthavatsalam et al. 2021, ARC-DA direct-answer, arXiv 2102.03315.
- Li et al. 2024, generator-validator consistency (ICLR 2024).
- Tsvilodub et al. 2024, predictions not robust under scoring-method variation, arXiv 2403.00998.
- Pacchiardi et al. 2024, simple features predict benchmark answers, arXiv 2410.11672.
- Balepur, Ravichander, Rudinger 2024, Artifacts or abduction (choices-only MCQA).
