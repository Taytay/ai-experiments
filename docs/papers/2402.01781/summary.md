# When Benchmarks are Targets: Revealing the Sensitivity of Large Language Model Leaderboards

- arXiv: 2402.01781 (v2, 2024) - https://arxiv.org/abs/2402.01781
- Authors: Alzahrani, Alyahya, Alnumay, Alrashed, Alsubaie, Almushayqih, Mirza, Alotaibi, Al-Twairesh, Alowisheq, Bari, Khan (NCAI / SDAIA)
- Venue: ACL 2024 (arXiv v2 comment: "updated with ACL 2024 camera ready version")
- Source: docs/papers/2402.01781/2402.01781.pdf, docs/papers/2402.01781/paper.txt

> Provenance: compiled from reading notes taken from the paper (docs/papers/2402.01781/notes.md) and the arXiv metadata; the PDF could not be rendered by the Read tool in the pass that produced this file. Re-verify any number against paper.txt before quoting it in the report.

## One-paragraph summary

Multiple-choice leaderboard rankings are fragile: on MMLU, changing the order of answer choices, replacing the A/B/C/D symbols with rare tokens, fixing the gold answer to one position, or switching from symbol scoring to cloze scoring moves models by up to 8 positions among 11 models (abstract; Figure 1). The paper runs three families of perturbations (choice format and order; prompt wording and scoring method; in-context knowledge manipulation) on 11 open models from 2.7B (phi-2) to 70B, and measures accuracy, the recall standard deviation across option positions (RStd, from Zheng et al. 2023) and rank agreement between perturbed and original leaderboards (normalised Kendall tau). It finds selection bias in every model; that token bias and position bias cannot be cleanly separated even with rare symbols; that models also exhibit scoring bias (rankings differ by scoring rule); that small prompt edits and few-shot changes are benign (tau_k > 0.9); and that models copy answers given in context even when those answers are wrong. Its recommendation is hybrid scoring: show the options in the prompt, then score the answer text (length-normalised) rather than the letter.

## Problem

Practitioners choose base models from MCQ leaderboards such as MMLU. If rank order depends on minute formatting choices (option order, symbols, scoring rule), the leaderboard is not informative. Prior work showed sensitivity to option order (Pezeshkpour and Hruschka 2023) and token/position bias in symbol selection (Zheng et al. 2023); this paper asks how much each perturbation changes the ranking and what an evaluator should do.

## Method

Three scoring schemes (Section 3.2, Figure 3):

- **Symbol scoring.** Prompt = question + lettered options. Scored on the letter tokens; prediction = option whose symbol has the highest likelihood. MMLU default in lm-evaluation-harness.
- **Cloze scoring.** Prompt = question only; options never shown. Each option's text is appended in turn; prediction = option with the maximum length-normalised likelihood. ARC default in the harness.
- **Hybrid scoring.** Prompt = question + lettered options, but scored on each option's text (not its letter), likelihood normalised by length.

Perturbations: (1) choice format and order (random and fixed reorderings; gold fixed to A/B/C/D; few-shot answers fixed to one position; A/B/C/D replaced by `$ & # @` or rare tokens); (2) prompt and scoring (drop subject name; `Correct Answer:`; symbol vs cloze vs hybrid); (3) in-context knowledge (few-shot examples that are trivial, out-of-domain, contain the correct answer, or contain a wrong answer).

Metrics: accuracy; RStd; tau_k = (tau + 1)/2 between rankings. Data: MMLU (14,042 items), a cleaned 3-subject subset for permutations, ARC-Challenge for the scoring comparison. Models: phi-2 (2.7B), Yi-6b, Yi-34b, Mistral-7b (+Instruct), Llama-2 7b/13b/70b (+chat). Nothing below 2.7B.

## Experiments and results

- **Order and position (5.1).** Shuffling choices moves 5 of 11 models, Yi-6b by 5 positions, tau_k = 0.564 (Table 1). Fixing the gold position gives tau_k 0.455 to 0.855 depending on position (Table 2, App. A.6).
- **Symbols (5.2).** Rare symbols lower accuracy for every model and usually raise RStd (Figure 4, Tables A.11, A.12). Token and position bias "difficult to mitigate" and not separable by symbol swaps.
- **Scoring bias (5.3, Figure 7, Tables A.13 to A.16).** MMLU zero-shot cloze: RStd falls to 1 to 4 (from 3 to 16 under symbol) but accuracy drops for every model (Yi-34B 73.4 to 49.3, phi-2 54.5 to 40.7, Llama-2-70b 65.4 to 48.7, Llama-2-7b 41.8 to 40.8); tau_k with the symbol leaderboard 0.527. Hybrid: RStd 2 to 5, smaller accuracy loss (Yi-34B 59.5, Llama-2-70b 55.1, Llama-2-7b 37.8); tau_k 0.709. On ARC-Challenge (cloze default, Table A.2) symbol raises accuracy for all but Llama-2-7b (Yi-34B 61.5 to 90.7) but raises RStd; hybrid beats cloze for most (Yi-34B 83.0, Llama-2-70b 72.6, phi-2 58.4), RStd 2.7 to 9.1, tau_k 0.782. Quoted: "Cloze scoring can essentially eliminate bias since the choices are never presented to the model, but LLMs tend to score poorly when using this method. This also does not reflect a true MCQ setting." Hybrid "represents an acceptable balance".
- **Benign perturbations (5.4).** Dropping the subject name or `Correct Answer:` changes accuracy < 1.5 points, tau_k >= 0.93 (Tables A.20 to A.23).
- **In-context knowledge (5.5).** Correct answer in a one-shot example: 61 to 99% accuracy (Table A.27); wrong answer: 4 to 37% (Table A.26). All five few-shot answers on one letter: 15 to 28 points lost (Table 3, A.28).
- **Scale.** Smallest models least stable: "rankings break down under slight perturbations, particularly in the medium to small model sizes".

## Limitations

- Cannot separate token bias from position bias, nor explain their origin; contamination not ruled out.
- Hybrid "still not completely robust to perturbations".
- MMLU-centric; 11 models, none below 2.7B.
- The length normalisation itself (mean per token vs summed vs byte-length) is not examined.
- Option-prior effects for invented or nonsense answer strings (this project's case) are not studied.

## Relevance to this workspace

Informs EVAL-1, EVAL-2, EVAL-3 directly; bears on EVAL-6, STAT-1, STAT-2, BASE-3.

- **EVAL-1.** The project's scorer (`experiments/exp_curriculum.py:89-108`) is cloze scoring in this paper's exact sense. The Timmy items, where candidate labels appear in the prompt as demo labels, are hybrid-like. The paper's finding that cloze has the lowest bias but lowest accuracy and "does not reflect a true MCQ setting" matches the observed 20.6 bare-format vs 100.0 trained-format recall gap.
- **EVAL-2.** RStd is computable only from per-item predictions, which the project does not save. Even cloze leaves RStd at 1 to 4, so option-content prior still matters; the paper offers no content-prior correction, so PMI (2104.08315) or PriDe (2309.03882) remain the tools.
- **EVAL-3.** Entirely likelihood-based; no free generation. Its "true MCQ setting" remark supports adding an options-listed (hybrid) condition as the bridge to generation.
- **EVAL-6 / STAT-1 / STAT-2.** Rank flips arise from 1 to 5 point accuracy differences, the same size as the project's headline gaps. Report rank stability across scorers, not one number.
- **BASE-3 / oracle context.** Models copy in-context answers indiscriminately (61 to 99% with the right answer, 4 to 37% with a wrong one), so oracle-context ceilings measure copying and format compliance, not knowledge.
- **Scale.** Qwen2.5-3B sits at the bottom of the studied range; 0.5B is below it. Expect more scorer sensitivity than reported.

### Concrete recommendation for this project's scorer

1. Add a hybrid scorer to every ladder level: list the options after the question, end at `Answer:`, score each option text with the existing length-normalised log-likelihood. Expect a large jump on bare L1 recall (same effect as `L1_recall_fmt`).
2. Do not switch to symbol (letter) scoring: RStd 8 to 16 for 7B and 13B models zero-shot (Table A.1).
3. Report three scorers on the same saved logits: cloze, hybrid, PMI-calibrated cloze. Persist per-item argmax and per-option scores so RStd, a predicted-label histogram and bootstrap CIs can be computed.
4. Report scorer agreement (tau_k-style rank agreement of arms across scorers); a conclusion that survives only under one scorer is a scoring artefact.
5. Keep a free-generation check as a separate line (EVAL-3).

## Key references worth following up

- Zheng et al. 2023, "Large Language Models Are Not Robust Multiple Choice Selectors", arXiv 2309.03882 (RStd; PriDe).
- Pezeshkpour and Hruschka 2023, order sensitivity, arXiv 2308.11483.
- Robinson, Rytting and Wingate 2023, arXiv 2210.12353 (symbol vs cloze accuracy).
- Zhao et al. 2021, "Calibrate Before Use", arXiv 2102.09690.
- Sclar et al. 2023, prompt-format sensitivity, arXiv 2310.11324.
- Holtzman et al. 2021, "Surface Form Competition", arXiv 2104.08315.
