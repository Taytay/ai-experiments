# Self-Tuning: Instructing LLMs to Effectively Acquire New Knowledge through Self-Teaching

- arXiv 2406.06326v5 (16 May 2025) - https://arxiv.org/abs/2406.06326
- Xiaoying Zhang, Baolin Peng, Ye Tian, Jingyan Zhou, Yipeng Zhang, Haitao Mi, Helen Meng (CUHK, Tencent AI Lab)
- Venue: preprint (arXiv v5)
- Source: docs/papers/2406.06326/paper.txt

## One-paragraph summary
Self-Tuning attacks the gap between memorizing documents (low perplexity) and being able to extract their facts at question time. It first teaches the model *how to learn* from documents by training on a held-out training corpus presented as plain text plus a set of self-supervised "self-teaching" tasks (summarize, list entities, NLI, teach, flashcards, fill-in-the-blank, MCQ, sentence completion) together with QA pairs; then it trains on the new test documents while reviewing the training QA; then on the test documents alone. On new Wikipedia pages (post-cutoff) with LLAMA2-7B, exact match on extraction rises from 3.62 (continued pretraining) and 11.61 (PIT) to 31.52, roughly matching the open-book ceiling of 31.83 (Table 3), while NQ and CommonsenseQA retention is preserved or improved. Gains replicate on Qwen2-7B, Mistral-7B, Gemma-7B, LLAMA2-13B and a news corpus.

## Problem
Continued pretraining on new documents lowers perplexity but the model cannot answer questions about them (Allen-Zhu and Li; Jiang et al. 2024c). Instruction tuning after CPT helps little; PIT (QA before documents) helps more but "underestimates comprehension".

## Method
- Datasets: Wiki-Newpages-2023-QA, first paragraphs (about 60 to 70 tokens) of new Wikipedia articles, with GPT-4-generated QA pairs and NLI items (Table 1: Wiki-Bio train 6,136 QA / 1,136 docs, test 663 QA / 127 docs; Wiki-Multi; Wiki-Film for cross-domain).
- Self-teaching tasks (Table 21), all template-based with spaCy/NLTK, no LLM: memorization (next-token on the document); comprehension (title generation, entity gist, NLI with a corrupted-entity negative); self-reflection ("Tell me about X" -> document, keyword flashcards -> document, cloze with one entity blanked, cloze MCQ with three in-document distractor entities, sentence completion after the last preposition). Task mix on Wiki-Bio (Fig. 8): NLI about 24%, sentence completion 16.5%, MCQA 13.5%, fill-in-the-blank 10.8%, the rest about 8.8% each.
- Three stages (Sec. 4.1, Table 2): (1) train docs with tasks plus train QA, 2 epochs; (2) test docs plus train QA review, 1 epoch; (3) test docs only, 2 epochs, with 128 replayed QA examples. All methods see the test docs for 3 epochs (CPT: 5).
- Losses: document loss averaged over all tokens (Eq. 4); QA loss averaged over answer tokens only (Eq. 5). LR 5e-6, batch 8, 8xV100 32 GB.
- Eval: PPL on test docs; open-ended generation scored by EM/F1/Recall/Rouge-L/entailment accuracy, 5-shot, temperature 1; NLI accuracy zero-shot; retention on NQ-open and CSQA.

## Experiments and results
- Wiki-Bio, LLAMA2-7B (Table 3): closed-book EM 2.87, open-book 31.83. CPT PPL 7.28 / EM 3.62; standard instruction tuning 5.13; PIT PPL 2.08 / EM 11.61; Self-Tuning PPL 1.11 / EM 31.52 / F1 50.83 / NLI 44.31; NQ EM 16.45 and CSQA 66.01 vs 16.05 and 53.40 for the untouched model.
- Multi-domain EM 16.51 vs PIT 8.72; cross-domain (train on Bio, test on Film) 16.44 vs 4.50 (Table 3).
- Variants (Table 4/12): without QA review EM 23.68; reading-comprehension format (Cheng et al.) 17.65; pre-review 25.94; full 31.52. Ablation (Fig. 4): removing self-reflection tasks hurts more than removing comprehension tasks.
- Dynamics (Fig. 3): PPL near 1 within 3 epochs; EM exceeds open-book from epoch 5 and peaks at epoch 25 (about 5 points above open-book); NQ EM drops only 2 to 3 points over 50 epochs.
- Other models (Table 5): Qwen2-7B EM 28.51 vs PIT 9.53; Mistral-7B 36.50 vs 23.08; WebNews corpus 28.74 vs 18.96. LLAMA2-13B 39.37 vs 19.61 (Table 13). LLAMA2-7B-CHAT 29.41 vs 13.12 but retention degrades; the authors recommend injecting into base models (App. J).
- Replay (Table 16): Self-Tuning + 500 replayed QA raises EM to 39.82, NQ EM to 22.67, CSQA to 73.55.
- Training on test docs with self-teaching QA pairs directly (Table 17) gives only EM 12.07; the staged "learn to learn" design matters.
- Cost (App. K, 8xV100): CPT 113 s, PIT 6,206 s, Self-Tuning 5,221 s.
- Error analysis (Table 11): 76% plain wrong answers, 10% paraphrase mismatches, the rest granularity.

## Limitations
- Documents are single 60-token paragraphs; the "corpus" is tiny per document, and each fact is a single sentence.
- Needs a training split of documents *with* GPT-4 QA pairs in the same style as the test QA; cross-domain transfer is shown but weaker.
- Evaluation decodes at temperature 1 and uses 5-shot prompts from the same source; three seeds are reported only as significance in App. T.
- Mixture fractions are by sequence; document loss is over all tokens and QA loss over answers, so the effective gradient weight is unreported (same confound as our TRAIN-1).
- 7B to 13B models only; no 0.5B to 3B results.

## Relevance to this workspace
- EVAL-1/EVAL-3: the 20.6 vs 100.0 bare-format vs trained-format recall gap is exactly the extraction problem this paper targets; its fill-in-the-blank, cloze-MCQ and sentence-completion tasks are cloze training in an autoregressive model, which is the MODEL-2 idea without switching to an encoder.
- TRAIN-1: our K/E/R mix is document text plus symbol-tuning episodes plus replay. Self-Tuning's E stream is *derived from the K texts themselves* (blanked attributes, completed sentences, MCQ over in-document entities), not only from label-induction episodes. Add a fourth stream of self-teaching items generated from the field-guide entries (no LLM needed) and sweep its fraction alongside E. Log label-token counts per stream, since the paper shares the per-token confound.
- TRAIN-2: their staging works because stage 2 reviews QA and stage 3 replays 128 QA items; our arm D has neither, which supports the "interleave or replay" reading.
- MODEL-1 extension: chat checkpoints retain worse (App. J); prefer base models for injection.
- Prediction: adding self-teaching cloze items over the K texts should lift bare `L1_recall` towards `L1_recall_fmt`, and NLI-style items may help L2/L3 manipulation; it will not by itself fix option-prior bias (EVAL-2).
- Cost on one 24 GB GPU: no generator needed; template tasks multiply K tokens by roughly 3 to 5. On Qwen2.5-3B LoRA at 851 tok/s, an arm with 3M to 4M tokens is 60 to 80 min plus 7 min eval; on 0.5B full FT it is minutes.

## Key references worth following up
- Jiang et al. 2024c, PIT, arXiv 2402.12847
- Cheng et al. 2024, Adapting LLMs via reading comprehension (ICLR)
- Saito et al. 2024, positional bias in knowledge extraction, arXiv 2402.12170
- Jiang et al. 2024a, Mix-CPT, arXiv 2407.10804
- Mecklenburg et al. 2024, arXiv 2404.00213
- Allen-Zhu and Li, arXiv 2309.14316
