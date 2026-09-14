# Tokenization Falling Short: On Subword Robustness in Large Language Models

- arXiv 2406.11687v3, 4 Oct 2024 (v1 June 2024) - https://arxiv.org/abs/2406.11687
- Yekun Chai, Yewei Fang (equal), Qiwei Peng, Xuhong Li (Baidu, ModelBest, University of Copenhagen)
- Venue: preprint (no venue line in the text); code/data at github.com/FloatAI/TKEval
- Source: docs/papers/2406.11687/paper.txt

## One-paragraph summary
The paper names the "curse of tokenization": subword tokenizers are sensitive to typos and length variation, treat capitalization variants as unrelated token ids, and give the model no view of the characters inside a token (Fig. 1: cosine between the embedding of "assignment" and "assign"+"ment" is 0.21; "import" vs "im"+"port" is 0.13). It probes Llama3-8B/70B, Mistral-7B, Mixtral-8x7B and GPT-4 Turbo on thirteen tasks in three groups: complex problem solving (anagrams, LaTeX theorem identification), token-structure probes (character counting, n-th character, case conversion, common substrings/subsequences), and typographically corrupted MMLU, TruthfulQA, GSM8K and HumanEval. Scale and few-shot demos help on the probes but every model, regardless of size, degrades under character-level noise. Post-training Mistral-7B with BPE-dropout at p = 0.2 improves the structure probes; higher rates hurt.

## Problem
Tokenization converts text into ids from a fixed vocabulary; a one-character change (a typo, a case flip) can produce an entirely different id sequence, and the embedding table has no built-in notion that "EL|RH|OL|M" and "El|r|holm" spell the same word. The paper lists three symptoms (Sec. 4): length unawareness, case insensitivity (capitalization changes token ids), and blindness to internal token structure. It asks whether scaling fixes this and whether subword regularization at training time helps.

## Method
Three research questions with purpose-built benchmarks (Sec. 3-5, Appendix C):
- RQ1: Cycled Letters and Word Unscrambling from BIG-bench; Identify Math Theorems (54 LaTeX problems, perplexity scoring over options).
- RQ2: intra-token probes CC, NC, NCR, CCV and inter-token probes CS, LCS, LCSeq, built from ~300 hand-collected words; 0-3 shot, exact match. Test sets 200 items per task (Appendix Table 2).
- RQ3: corrupt only the question text of MMLU (14,042), TruthfulQA (817), GSM8K (1,319, 5-shot no CoT) and HumanEval (164). Character-level permutation within word boundaries at n-gram sizes 2/3/5 with 50% probability; character-level noise (insert/delete/replace at 10% each); token-level permutation and token-level noise (30%). MMLU/TruthfulQA scored by perplexity; temperature 0 elsewhere.
- Mitigation (Sec. 6): post-train Mistral-7B on 111,070 intra-token plus 14,400 inter-token synthetic examples with BPE-dropout p in {0, 0.2, 0.4, 0.6, 0.8}; 5 epochs, batch 16, 4096-token packing, AdamW (0.9, 0.95), peak LR 5e-5 to 1e-6, cosine, 10% warmup.

## Experiments and results
- Anagrams (Fig. 2, Fig. 3): Llama3-70B beats 8B on all length buckets, both collapse on 12-18-character words; extra demos do not monotonically help. GPT-4 Turbo is best everywhere.
- LaTeX theorems (Table 1): Llama3-70B 62.26 zero-shot, 79.25 one-shot, 69.81 two-shot, 71.70 three-shot; Llama3-8B 41.51/45.28/45.28/35.85; Mixtral-8x7B 49.10/56.60/64.20/62.30 vs Mistral-7B 47.20/43.40/37.70/37.70.
- Intra-token probes (Fig. 4): Llama3-8B goes from 0% (0-shot) to 81% (3-shot) on character counting; Llama3-70B from 1% to 55% on n-th character; GPT-4 Turbo tops out at 52% one-shot on reverse n-th character. Few-shot demos matter more than scale on these.
- Inter-token probes (Fig. 5): LCSeq is hard for everyone; Llama3-8B improves only from 1% to 4% with three shots.
- Typographical variation (Fig. 6): across all datasets and models, noise (insert/delete/replace) hurts far more than reordering; character-level corruption hurts more than token-level; larger n-gram windows (5) hurt less than 2; "all models experienced evident performance degradation, regardless of the parameter sizes." Deltas are shown only graphically; clean baselines are printed (e.g. MMLU Llama3-8B 62.14, Llama3-70B 75.43; HumanEval Llama3-70B 67.07, GPT-4 Turbo 88.41).
- BPE-dropout (Fig. 7, Fig. 8, Appendix D): p = 0.2 "frequently surpasses the baseline", consistently on CCV, NC and LCS; p = 0.6/0.8 lower on most tasks, attributed to too few epochs; moderate rates (0.2-0.4) help most in zero-shot. CS and CC are robust to dropout; NC, LCS, NCR degrade most at high rates.

## Limitations
- The case symptom is asserted (Sec. 4 item B) and the CCV probe tests whether a model can *convert* case, but no experiment feeds uppercased inputs to a downstream task, so the paper gives no number for the accuracy cost of ALL-CAPS text.
- BPE-dropout is evaluated only on the RQ2 structure probes, never on the RQ3 typo benchmarks, so there is no evidence that it repairs MMLU/GSM8K robustness.
- RQ3 results are plots without tables; Sec. 6 says 111k training examples while the Limitations section says ~30k; the CCV example in Appendix A.2 duplicates the CC example.
- Single 7B model and 5 epochs for the mitigation; randomness of dropout tokenization is itself acknowledged as a source of variance.

## Relevance to this workspace
- MODEL-4: the paper's framing supports the reviewer's reading. Case changes are a *token-identity* change, not a semantic one, and models have no mechanism to relate the two id sequences unless training showed both. Our measured 12.6% token survival under uppercasing is a sharper, more direct version of the paper's argument than anything the paper itself quantifies. Scale does not fix this: even GPT-4 Turbo and Llama3-70B degrade under character noise (Sec. 5).
- Mitigation for bank strings (DATA-4, BASE-5, TRAIN-4): (a) render each merchant's training texts in several cases and truncations so that both id sequences co-occur with the same facts; (b) apply BPE-dropout (p about 0.2, the paper's sweet spot) to the *name spans* of knowledge texts during LoRA training, which yields diverse segmentations of "Elrholm" and should make "EL|RH|OL|M" less alien. HF `tokenizers` BPE models expose a `dropout` field, so this is a tokenizer-config change, not new code. Expect the paper's caveat: benefit is shown on structure probes, so verify on `bank_category` directly.
- MODEL-3: tokenizer coverage differs by family (Llama3 128k vocab vs Mistral 32k, Appendix B.1); when adding a non-Qwen family, record the same token-survival statistic.
- EVAL-3: the paper's generation-based EM scoring with temperature 0 is the pattern for the missing free-generation check.

## Key references worth following up
- BPE-dropout: Provilkov et al. 2020, arXiv 1910.13267
- Subword regularization: Kudo 2018, arXiv 1804.10959
- Cao et al. 2023, "Unnatural error correction" (scrambled-text robustness of GPT-4), EMNLP 2023
- ByT5 (Xue et al. 2022), arXiv 2105.13626; PIXEL (Rust et al. 2023), arXiv 2207.06991 (tokenizer-free alternatives)
- BIG-bench (Srivastava et al. 2022), arXiv 2206.04615 (source of the anagram and theorem tasks)
