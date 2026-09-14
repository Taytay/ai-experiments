# Surface Form Competition: Why the Highest Probability Answer Isn't Always Right

- arXiv 2104.08315 (v9, 20 Nov 2022; first version 2021) - https://arxiv.org/abs/2104.08315
- Ari Holtzman, Peter West (equal), Vered Shwartz, Yejin Choi, Luke Zettlemoyer (UW / AI2)
- Venue: EMNLP 2021 (pages 7038-7051, as cited in the OLMES reference list)
- Source: docs/papers/2104.08315/paper.txt

## One-paragraph summary
When a causal LM answers multiple choice by scoring each option string, the options compete for probability mass not only with each other but with every other surface form of the same concept ("Bathtub", "A bathtub", "bath tub"). Rare or unusual strings therefore lose even when they are the intended answer. The paper proposes Domain Conditional PMI (PMI_DC): score an option by how much more probable it becomes given the question than given a short "domain premise" alone, i.e. log P(y|x) minus log P(y|x_domain). Across 16 splits of 13 datasets and all GPT-2 (125M-1.6B) and GPT-3 (2.7B-175B) sizes, PMI_DC beats raw likelihood (LM), mean-per-token likelihood (AVG), the unconditional baseline (UNC) and Zhao et al.'s contextual calibration (CC) on most datasets (Tables 1, 2, 6). A "COPA Flipped" experiment, where the fixed string is the premise and the options are the contexts, removes surface form competition and makes LM, AVG and PMI_DC identical (Table 5), which the authors take as evidence that competition is the mechanism behind the gap.

## Problem
Zero-shot MCQ scoring is usually argmax_i P(y_i | x) (LM) or the length-normalised AVG (mean token log-prob). Both measure "how likely is this exact string", not "which concept answers the question". Generic strings ("I don't know"), common paraphrases and short or frequent forms receive prior mass that has nothing to do with the question, so the ranking is a mixture of knowledge and surface-form prior. The paper wants a scoring rule that isolates the question's contribution.

## Method
- LM: argmax P(y_i|x). AVG: argmax (1/|y_i|) sum_j log P(y_ij | x, y_i<j) (Section 3.1).
- PMI_DC (Section 3.2): exp PMI_DC(x, y, domain) = P(y | x, domain) / P(y | domain), estimated as P(y|x) / P(y|x_domain), where x_domain is a short domain premise, "usually just the ending of the conditional premise x" (for COPA "because"/"so"; for ARC, OBQA, CQA "the answer is:"; for BoolQ "answer:"; for AG News "topic:"; for SST "[The quote] has a tone that is"; Table 7). Computed as the difference of two summed log-probabilities. Because PMI is symmetric, the argmax equals argmax P(x | y_i, domain), i.e. "scoring by premise", which is free of competition between options.
- UNC: argmax P(y_i | x_domain), a sanity check that the question is being used at all (Section 3.3).
- CC: contextual calibration of Zhao et al. 2021, reported where available.
- No fine-tuning; templates in Appendix B; code released.

## Experiments and results
- GPT-3 zero-shot (Table 1). OBQA 2.7B: UNC 10.0, LM 17.2, AVG 27.2, PMI_DC 42.8; at 175B 33.2 / 43.8 / 58.0. CQA 2.7B: LM 33.2, AVG 36.0, PMI_DC 44.7; 175B 61.0 / 57.4 / 66.7. ARC-Challenge 2.7B: 21.6 / 25.5 / 30.5. SST-2 2.7B: LM 53.7, PMI_DC 72.3, CC 71.4. TREC 2.7B: LM 29.4, AVG 19.2, PMI_DC 57.2, CC 38.8.
- GPT-2 (Table 6): gains at every size, e.g. OBQA 125M LM 0.164, AVG 0.272, PMI_DC 0.324; CQA 125M 0.255 / 0.307 / 0.364.
- Table 2: PMI_DC is best or tied on the majority of splits at every model size; the smallest margin over the runner-up (AVG at GPT-3 175B) is "over 40 percentage points" of splits.
- Robustness (Table 3): over the 15 SST-2 templates from Zhao et al., PMI_DC has the highest mean at every size.
- Few-shot (Table 4, 4-shot): PMI_DC generally wins; LM beats it for two sizes on SST-2 (6.7B: 92.9 vs 79.8).
- Failure cases (Section 6): HellaSwag prefers AVG (internal coherence of the continuation matters more than premise-hypothesis fit); ARC-Easy prefers LM (stock, a priori likely answers); BoolQ prefers UNC (nothing beats majority except PMI_DC at 175B).
- Why AVG works (Section 6): BPE tokens have roughly uniform unigram frequency, so token count is a crude unigram log-prob estimate; length normalisation is therefore an implicit, weaker unconditional correction.
- COPA Flipped (Table 5): LM = AVG = PMI_DC on the flipped data at every size, and log P(y|x) for two paraphrased correct answers differs by ~4 nats under LM but is stable under scoring-by-premise (Section 5.2).

## Limitations
- Zero-shot, pretrained GPT models only; no fine-tuned models, so the paper says nothing about how a learned prior (as in this project's arms) interacts with PMI_DC.
- The domain premise is hand-chosen per dataset; PMI_DC is sensitive to that choice and the paper does not sweep it.
- PMI_DC still scores options independently; it "does not go far enough" when options interact ("all of the above") (Section 7).
- No confidence intervals; single template per dataset for the main table.
- Doubles forward passes (one unconditional pass per option).

## Relevance to this workspace
- EVAL-1, EVAL-2: the project scorer (`exp_curriculum.py:106-108`) is exactly AVG. PMI_DC is the missing calibrated scorer named in QUESTIONS.md. Exact formula for this project: score_i = sum_j log P(o_ij | prompt, o_i<j) minus sum_j log P(o_ij | x_domain, o_i<j), summed (not averaged) over the option tokens; argmax over i. Domain premise: either the bare suffix "Answer:" (the OLMES choice) or, closer to Holtzman's practice of keeping the task domain, a content-free copy of the template ("Question: What type is this creature?\nAnswer:", "Question: Which group does this creature belong to?\nAnswer:"). Report both; the second keeps "type-name given a type question" as the divisor rather than "type-name given nothing".
- L5 novel choices (8.8-17.5 vs chance 12.5): options are common English synonyms with very different priors; this is the textbook case (OBQA "Whirlpool bath" vs "Bathtub") where PMI_DC gains 15-25 points at 2.7B.
- UNC is the partial-input baseline this project lacks: score every level with x_domain only. Any level where UNC beats chance has an artifact, which is the mechanism behind "below chance" and the 42.7% base score on unknowable held-out induction (EVAL-6).
- Timmy items: nonsense labels sampled from NONSENSE have different subword priors; PMI_DC is "naturally applicable to datasets where the set of valid answers varies between questions". Because the labels also appear in the demos, the domain premise should keep the demos and blank the query name (content-free query, as in CC) rather than drop the demos.
- Models below 7B: GPT-2 125M-1.6B and GPT-3 2.7B all benefit, often more than larger models (Table 6, Table 1).
- Cloze format: the whole paper is cloze scoring; options are never listed.

## Key references worth following up
- Zhao et al. 2021, Calibrate Before Use, arXiv 2102.09690 (content-free calibration; the CC baseline).
- Brown et al. 2020, GPT-3, arXiv 2005.14165 (origin of AVG and of the "A:" unconditional trick).
- Schick and Schuetze 2020, arXiv 2001.07676 (cloze/PET lineage, ties to MODEL-2).
- Perez, Kiela, Cho 2021, True Few-Shot Learning, arXiv 2105.11447 (validation-set leakage in prompt selection).
- Jiang et al. 2020, How Can We Know When Language Models Know, arXiv 2012.00955 (calibration of LM QA).
