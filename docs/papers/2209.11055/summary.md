# Efficient Few-Shot Learning Without Prompts (SetFit)

- arXiv 2209.11055v1, 22 Sep 2022 - https://arxiv.org/abs/2209.11055
- Lewis Tunstall, Nils Reimers, Unso Eun Seo Jo, Luke Bates, Daniel Korat, Moshe Wasserblat, Oren Pereg (Hugging Face, cohere.ai, UKP Lab TU Darmstadt, Intel Labs)
- Venue: preprint (v1); code github.com/huggingface/setfit
- Source: docs/papers/2209.11055/paper.txt

## One-paragraph summary
SetFit is a two-step, prompt-free few-shot text classifier built on Sentence Transformers. Step one fine-tunes the ST contrastively on sentence pairs generated from the few labeled examples (same class = similar, different class = dissimilar); step two fits a logistic-regression head on the resulting embeddings. With 8 labeled examples per class, a 110M-parameter SetFit-MPNet averages 62.3 across six datasets, 19.3 points above standard RoBERTa-large fine-tuning and on par with the 3B-parameter T-Few, while training in about 30 seconds on a 16 GB GPU. On the RAFT benchmark SetFit-RoBERTa (355M) scores 71.3, above GPT-3 (62.7) and PET (69.6) and above the human baseline on 7 of 11 tasks.

## Problem
Few-shot methods of 2022 (in-context GPT-3, PEFT such as T-Few, cloze/prompt methods such as PET and ADAPET) need billion-parameter models, hand-written prompts and verbalizers, and show high variance across prompts. Practitioners want something that trains on a laptop-class GPU from a handful of labeled sentences.

## Method
(Sec. 3.1, Fig. 2) Given K labeled examples D = {(x_i, y_i)} over classes C:
1. ST fine-tuning. For each class c, sample R positive pairs (x_i, x_j, 1) with both from c, and R negative pairs (x_i, x_j, 0) with x_i from c and x_j from another class; |T| = 2R|C| pairs; R = 20 by default. Train the ST body with cosine-similarity loss, LR 1e-3, batch 16, max length 256, 1 epoch (Sec. 4.4). The pair construction inflates K examples into up to K(K-1)/2 distinct pairs.
2. Head. Encode the original K examples with the fine-tuned ST and fit logistic regression on (embedding, label).
3. Inference: prediction = head(ST(x)).
Bodies (Table 1): all-roberta-large-v1 (355M), paraphrase-mpnet-base-v2 (110M), paraphrase-MiniLM-L3-v2 (15M). Baselines: RoBERTa-large fine-tuning, ADAPET (albert-xxlarge-v2), PERFECT, T-Few 3B (5 seeds, median). Ten random training splits per dataset and size; mean and sd reported.

## Experiments and results
- Table 2 (N = 8 per class, six-dataset average excluding AGNews): FINETUNE 43.0 +/- 5.2, PERFECT 48.7 +/- 6.0, ADAPET 58.3 +/- 3.6, T-Few 3B 63.4 +/- 1.9, SetFit-MPNet 62.3 +/- 4.9. At N = 64: FINETUNE 69.7 +/- 7.8, PERFECT 72.7 +/- 1.9, ADAPET 73.8 +/- 2.2, T-Few 3B 70.3 +/- 1.5, SetFit-MPNet 75.3 +/- 1.3. Full-data fine-tuning average 84.8. SetFit's margin over FINETUNE shrinks from 19.3 to 5.6 points as N grows; over ADAPET from 4.0 to 1.5.
- RAFT (Table 3, leaderboard Sept 2022): T-Few 11B 75.8, human 73.5, SetFit-RoBERTa 71.3, PET 69.6, SetFit-MPNet 66.9, GPT-3 62.7.
- Multilingual MARC (Table 4, MAE x 100, N = 8): SetFit beats FINETUNE and ADAPET in all of each/en/all settings; best when trained on English only.
- Distillation (Fig. 3): a SetFit-MiniLM student beats a standard MiniLM student by 24.8, 25.1 and 8.9 points on AGNews, Emotion and SST-5 with 8 unlabeled examples; parity at 1K.
- Cost (Table 5, Sec. 7.2): SetFit-MPNet is 19x cheaper than T-Few 3B in FLOPs (score 62.3 vs 63.4); SetFit-MiniLM 123x cheaper at 60.3. Training on N = 8 takes about 30 s on a p3.2xlarge (16 GB) vs about 700 s on a 40 GB-plus instance for T-Few 3B; checkpoints are 70 MB and 420 MB vs 11.4 GB.

## Limitations
- No ablation isolating the contrastive body fine-tuning from the head: there is no "frozen ST + logistic regression" row and no nearest-centroid or kNN row, so the paper cannot say how much of the gain is step 1.
- Only N = 8 and N = 64 per class; datasets are balanced sentiment/topic sets with 2-6 classes; variance can be large (Emotion at N = 64: 65.0 +/- 17.2).
- T-Few 11B was not run; AGNews excluded from the average because it is in T-Few's training data; SST-5 inputs overlap Rotten Tomatoes in T0's training.
- R = 20, one epoch, and cosine loss were tuned on dev sets and not swept in the paper.
- Preprint v1; the released library changed defaults later (not covered by this text).

## Relevance to this workspace
- BASE-4 (missing baseline): SetFit is the canonical few-shot embedding classifier and the right reference point for the 63-85% prototype result. Our prototype classifier is nearest-centroid on embeddings from an encoder that was contrastively trained on *name-to-attribute-text* pairs, never on user labels. SetFit differs in two ways that should be run as separate rows on the same items: (1) a logistic-regression head instead of centroids on the *same* embeddings (frozen base MiniLM and our fine-tuned MiniLM), and (2) an extra label-supervised contrastive pass over the user's K examples (2R|C| pairs, R = 20, cosine loss, LR 1e-3, one epoch), then the head. With K = 1 per class step 1 has no positive pairs, so SetFit needs K >= 2; report k in {2, 3, 10} to match REAL-4. Nearest-centroid is itself a linear classifier with tied covariance, so the LR-vs-centroid gap measures how much per-dimension reweighting buys at small K. Expect SetFit's own pattern: large gains over the frozen baseline at K = 8, shrinking by K = 64.
- REAL-4 (imbalanced 10-30 categories): SetFit's pair sampler is per-class, so it is balanced by construction; use it as the comparison arm when categories are imbalanced, and note the paper's high split-to-split variance argues for the 10-split protocol (STAT-1/STAT-2 for the embedding side).
- MODEL-2 / LIT-1: at N = 8 the prompt-free method beats the cloze-based ADAPET by 4.0 and PERFECT by 13.6 (Table 2), evidence that a masked-LM arm F is not automatically the better few-shot extractor for personal categories.
- MODEL-3: SetFit's body is swappable (Sec. 6 switches to multilingual MPNet), so the same baseline runs on gte-modernbert-base or Qwen3-Embedding-0.6B.
- Practical: SetFit-MPNet trains in ~30 s on 16 GB, so the whole baseline grid is minutes on the RTX 3090.

## Key references worth following up
- T-Few (Liu et al. 2022), arXiv 2205.05638
- PERFECT (Karimi Mahabadi et al. 2022), arXiv 2204.01172
- ADAPET (Tam et al. 2021), arXiv 2103.11955; PET (Schick and Schütze), arXiv 2001.07676
- Sentence-BERT (Reimers and Gurevych 2019), arXiv 1908.10084
- RAFT benchmark (Alex et al. 2021), arXiv 2109.14076
- Revisiting few-sample BERT fine-tuning (Zhang et al. 2021), arXiv 2006.05987
