# OLMES: A Standard for Language Model Evaluations

- arXiv 2406.08446 (v2, 11 Feb 2025; v1 June 2024) - https://arxiv.org/abs/2406.08446
- Yuling Gu, Oyvind Tafjord, Bailey Kuehl, Dany Haddad, Jesse Dodge, Hannaneh Hajishirzi (AI2 / UW)
- Venue: Findings of NAACL 2025 (as cited in the Balepur et al. reference list)
- Source: docs/papers/2406.08446/paper.txt

## One-paragraph summary
OLMES is a fully specified recipe for multiple-choice evaluation of base LMs: fixed prompt formats ("Question: ... Answer:"), five manually curated shots, a per-task rule for which log-likelihood normalisation to use in the cloze formulation (CF), and the rule to run both CF and the letter-based multiple-choice formulation (MCF) and report the better one. The normalisation choice is settled empirically over 15 base models from 1B to 70B: PMI normalisation for ARC-Challenge, CommonsenseQA and OpenBookQA, character normalisation for ARC-Easy, HellaSwag, MMLU, PIQA and Social IQa, no normalisation for BoolQ and WinoGrande (Tables 2, 3, 10-12). The paper documents why small models need CF (MCF is at chance until a model learns the letter format, around 400B tokens for OLMo-7B-0424, Figure 1) and why strong models need MCF (Llama3-70B MMLU 79.8 MCF vs 60.7 CF, Table 6).

## Problem
The same model on the same task is reported with 10-30 point spreads across papers because shots, prompt format, normalisation and formulation are unspecified (Table 1, Table 14: Llama3-8B ARC-Challenge 60.2 CF on the HF leaderboard vs 78.6 MCF in the model card). Base models below roughly 7B, or early in training, cannot answer with a letter at all, so a standard has to accommodate both regimes.

## Method
- CF: prompt ends at "Answer:", each option is substituted and scored by its token probabilities; options are never shown (Section 2.1). MCF: options listed as " A. <choice>" with a leading space so the label token matches the answer token (Appendix C.3); score the letter.
- Four CF normalisations (Section 3.3): none = ln P(a|q); token = ln P(a|q) / num_tokens(a); character = ln P(a|q) / num_characters(a), including the leading space; pmi = ln P(a|q) / P(a|u) with u = "Answer:".
- 15 models, 10 tasks, 1000-instance cap when a set exceeds 1500 items, 5 curated shots, 2048-token limit, default precision, macro average for MMLU.

## Experiments and results
- Normalisation win rates across 15 models (Table 3): ARC-Challenge pmi 66.7%, char 33.3%; OBQA pmi 100%; CSQA pmi 53.3%; MMLU pmi 53.3% vs char 46.7% (char chosen for cost); HellaSwag char 100%; WinoGrande none 100%; BoolQ none/char 46.7% each.
- Size of the effect for the smallest models (Tables 10-12): OBQA Pythia-1B none 20.2, char 28.6, token 30.4, pmi 40.4; OLMo-1B 26.0 / 33.0 / 38.4 / 47.6; TinyLlama-1.1B 24.4 / 34.8 / 35.8 / 45.0. ARC-Challenge Pythia-1B 26.1 / 28.4 / 29.0 / 31.4. PMI hurts where options are natural continuations: ARC-Easy average 70.0 pmi vs 78.7 char; HellaSwag 61.2 vs 74.5; PIQA 64.1 vs 77.6.
- Rationale: pmi is chosen where "answer choices tend to contain unexpected words or phrases that are less likely for models to generate"; BoolQ gets none because yes/no are single tokens and "models should be capable of producing such common words" without correction (Section 3.3). Character over token: the tokenizer-dependence argument "does not seem like a relevant argument" when the model is fixed; for long or similar-length answers the two differ little (Appendix C.2).
- CF vs MCF (Table 6, Figure 2): Pythia-1B MMLU 26.5 MCF vs 31.1 CF, ARC-Challenge 24.1 vs 31.4; OLMo-1B ARC-Challenge 25.3 vs 38.6. The weakest 8 of 15 models are at chance under MCF on ARC-Challenge. Llama3-70B ARC-Challenge 93.7 MCF vs 69.0 CF.
- Hybrid (options listed, answer text scored): once the first tokens disambiguate the option "most tokens ... would have probability near one"; it "usually scores in between the CF and MCF approaches" and is not adopted (Appendix C.2.2).
- Stability (Table 5): three benign prompt/shot variants move scores by at most 1.4 points; standard errors 0.8-2.2 on 500-1000 items. Floating-point ties can flip near-equal options (Section 3.5).
- Extended table (Table 13) includes Qwen2-0.5B (ARC-Challenge 48.4, MMLU 45.3), but the MCF/CF marker is not legible in the text extraction.

## Limitations
- Base models only, 1B and up in the main study; the 0.5B regime is only in the appendix.
- The normalisation rule is empirical per task, not derived; the paper explicitly declines to explain when token vs character matters.
- No treatment of fine-tuned models whose option priors were shaped by training.
- 5-shot throughout; zero-shot CF (this project's setting) is not benchmarked.
- No per-item outputs or bias metrics (RStd) are reported.

## Relevance to this workspace
- EVAL-1: the project scorer is OLMES CF with token normalisation, zero-shot. OLMES's evidence that CF is the only informative formulation for 1B-class base models supports keeping cloze as the primary score for Qwen2.5-0.5B, while the 3B model may already be in the regime where listing options helps; run both and report the max, as OLMES does.
- EVAL-2, which normalisation: for the L1/L5 option sets (fixed vocabularies of type names, categories, synonyms with unequal priors) OLMES's rule says pmi with u = "Answer:"; the prompt already ends in "Answer:", so the unconditional pass is cheap (8-12 short strings per level, cacheable once per model). For L2 yes/no use none (single tokens " Yes"/" No"). For L1_recall_fmt, where every option shares the sentence frame and differs only in the type word, none is also correct (the shared tokens contribute an identical offset). Character normalisation including the leading space is the harness-standard fallback and should be reported next to token normalisation.
- EVAL-3: OLMES defers generative tasks to future work; no help there.
- STAT-1/STAT-2: report a standard-error column per level as in Table 5; treat near-ties as ties (Section 3.5) or persist the margin already computed in `accuracy()`.
- Below 7B and cloze: Figure 1 and Table 6 are the strongest published evidence that MCF letters are uninformative for small base models and cloze is the right elicitation; the 20.6 vs 100.0 format gap is a prompt-format effect on top of that.

## Key references worth following up
- Gao 2021, Multiple choice normalization in LM evaluation (EleutherAI blog; origin of acc vs acc_norm).
- Biderman et al. 2024, Lessons from the trenches, arXiv 2405.14782.
- Robinson, Rytting, Wingate 2023, arXiv 2210.12353 (MCF vs cloze).
- Wiegreffe et al. 2023, Increasing probability mass on answer choices does not always improve accuracy (EMNLP 2023; hybrid results).
- Khatun and Brown 2024, arXiv 2401.07955 (small-model MCQ limitations).
- Sclar et al. 2023, arXiv 2310.11324 (format sensitivity).
