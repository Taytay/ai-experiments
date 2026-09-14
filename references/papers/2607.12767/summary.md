# Accuracy and Normalized Accuracy under Length Bias: Analysis, Guidelines, and a Bayesian Alternative

- arXiv 2607.12767 (v1, 14 Jul 2026) - https://arxiv.org/abs/2607.12767
- Koen Oostermeijer (Aleph Alpha Research)
- Venue: ICML 2026 (PMLR 306), printed in the paper header
- Source: references/papers/2607.12767/paper.txt

## One-paragraph summary
Likelihood-based MCQ scoring is length-biased in both directions: summed log-probabilities fall roughly linearly with completion length, so unnormalised scores favour short options, while dividing by token or byte count over-corrects and favours long ones. The paper defines length bias as the mean within-item Kendall tau between option length and score, measures it for 17 base and instruct models from 135M to 70B on eight benchmarks, and shows normalised accuracy is often more biased than the raw score it was meant to fix (Tables 1, 2, 4). It proposes Bayesian accuracy: subtract b times the option length from the summed log-probability, with b fitted from within-question centred regression on the evaluation set itself (Eq. 23-24, Algorithm 1). This needs no extra forward passes and brings |tau| to 0.03-0.07, several times lower than PMI or ANPMI (Table 4).

## Problem
Sum-scoring accumulates per-token loss, so even a "known" answer loses to a shorter distractor. The standard fix (mean per token or per byte) rescales signal and noise together and, empirically, flips the sign of the bias. Neither rule has a stated regime of validity, and PMI-style corrections double the compute and leave residual bias.

## Method
- Bias metric: for each item, Kendall tau_b between option lengths and scores (Eq. 14); average over items where defined (Eq. 16). Byte length in the main text, tokens in Appendix A.9.
- Scoring rules compared: standard (Eq. 6), token- and byte-normalised (Eq. 7-8), PMI (Eq. 9, sum-based), ANPMI (Eq. 10, PMI divided by minus the unconditional log-prob, from Cho et al. 2025), and Bayesian.
- Bayesian accuracy: model l(c|x) = a + b n + noise (Eq. 17-18), which corresponds to an exponential length prior P_prior(n) proportional to exp(-b n) (Eq. 22); the debiased score is l(c|x) - b n (Eq. 23). b is estimated by pooling within-question centred covariances of length and log-likelihood over the eval set (Eq. 24), which cancels item difficulty offsets; shrink toward 0 or pool templates when within-item length variation is small (Section 4.3).
- Benchmarks: ARC, ARC German, HellaSwag, MMLU Full-Text (options listed, full answer text scored), MMLU Cloze (options removed), OpenBookQA, SciQ, WinoGrande; zero-shot main, few-shot in Appendix A.6.

## Experiments and results
- Figure 1: total log-likelihood decreases approximately linearly with byte length on every benchmark except MMLU Full-Text, with a positive intercept at length one (the first option token is the hardest).
- Standard accuracy (Table 1): tau strongly negative on HellaSwag (-0.46 to -0.61), MMLU Cloze (-0.22 to -0.47), OpenBookQA (-0.16 to -0.35), ARC (-0.12 to -0.27); Qwen3 600M: ARC -0.20, MMLU Cloze -0.38, OBQA -0.30. Bias shrinks with size within a family but stays non-zero; Pythia 410M and SmolLM 135M are worst. MMLU Full-Text is near zero because once the first tokens pick an option "the remainder of the answer is largely a deterministic copy of text already appearing in the prompt" (Section 3.3.2).
- Byte-normalised (Table 2): tau uniformly positive, e.g. Llama 3.2 1B ARC 0.31, OBQA 0.36, WinoGrande 0.44; instruct models worse (Qwen3 4B Instruct ARC 0.43). Normalisation "introduces a strong length bias where there was little before" on ARC, OBQA, SciQ, WinoGrande.
- Rules of thumb (boxed): standard accuracy is safe when the answer text already appears in the prompt and the first tokens fix the option, or when all candidates have similar length; for normalised accuracy "the results do not reveal a simple heuristic" and it should be used "only once low length bias has been empirically confirmed".
- Bayesian (Table 3): almost all |tau| <= 0.1. Mean |tau| per model (Table 4, first row Llama 3.2 1B): standard 0.24, normalised 0.32, PMI 0.16, ANPMI 0.11, Bayes 0.04; the text summarises Bayes at 0.03-0.07, a 4x-8x reduction relative to normalised accuracy. Token-length version (Table 10): normalised 0.33-0.48 vs Bayes 0.03-0.06.
- PMI has its own positive length bias (Table 9: ARC 0.07-0.13; MMLU Full-Text 0.40-0.49); ANPMI is negative on HellaSwag (-0.22 to -0.31, Table 8).

## Limitations
- The correction only removes the linear length trend; it "does not remove other answer priors such as preferences for frequent words, phrases, or syntactic forms" (Section 4.2), which is precisely the surface-form prior.
- b is one global scalar per model-dataset-prompt; the paper admits per-item slopes vary (variance grows quadratically in length, Eq. 19).
- b is fitted on the test items themselves (label-free, but transductive); no held-out study of how noisy b is at n = 160.
- Accuracy numbers are not the object of study; only bias is reported, so the paper does not show whether debiasing raises accuracy.
- Zero-shot main results; one exemplar set for few-shot.

## Relevance to this workspace
- EVAL-2 directly: the reviewer's claim that mean-per-token "favours multi-token options because their later tokens are near-deterministic" is this paper's Figure 2 and Table 2 finding, measured on Qwen3 600M-4B. The project's scorer is the over-correcting rule; category options with 1-3 tokens ("Gas & Auto" vs "Rent") and type names with 2-4 tokens are exactly the within-item length variation that triggers it.
- Recommended scorer: compute the Bayesian score from the per-token log-probs already produced by `option_scores` (sum minus b times token count, or byte count); estimate b per level with Algorithm 1 over the 160 items. Zero extra forward passes. Report the within-item Kendall tau under mean, sum and Bayes as a diagnostic column; a level whose tau is far from zero has a length confound regardless of accuracy.
- Levels where length is provably irrelevant (rule of thumb): L2 yes/no (single tokens; use sum) and L1_recall_fmt (shared frame, "locally determined").
- Fixed option sets are a caveat: in L1 every item has the same 8 type strings, so length and identity are confounded and b will absorb part of the content prior; that is why PMI_DC and Bayes should be reported side by side rather than one replacing the other.
- Scale: Qwen3 600M, 1.7B, 4B are in every table, bracketing the project's 0.5B and 3B models.
- Cloze: "MMLU Cloze" (options removed) is the project format and shows the largest standard-scoring bias after HellaSwag; "MMLU Full-Text" is the hybrid format and shows why listing options removes the length problem.

## Key references worth following up
- Cho, So, Lee 2025, ANPMI, arXiv 2502.18798.
- Biderman et al. 2024, arXiv 2405.14782 (harness normalisation choices).
- Askell et al. 2021, arXiv 2112.00861 (early PMI-style scoring).
- Qwen Team 2025, Qwen3 technical report, arXiv 2505.09388.
