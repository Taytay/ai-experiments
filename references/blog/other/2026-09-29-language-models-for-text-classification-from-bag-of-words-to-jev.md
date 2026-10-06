# Language Models for Text Classification: From Bag-of-Words to Jev

*Source: https://magazine.sebastianraschka.com/p/classifier-history-and-jev  
Author: Sebastian Raschka, PhD (Ahead of AI)  
Subtitle: A Visual Guide to Bag-of-Words, RNNs, CNNs, Transformers, Jev-like APIs, and Calibration  
Published: 2026-09-29 (with two later in-post updates)  
Fetched: 2026-10-06 via the Substack API (`/api/v1/posts/classifier-history-and-jev`, `body_html`; audience: everyone, ~7,900 words)*

> This file is a section-by-section digest, not a verbatim copy: the article is copyrighted, so the prose is paraphrased,
> quotes are kept short, and every number, table and code example the author reports is kept. Read the original at the
> source link for the full text. Figures are linked to their original images with the author's captions (paraphrased where long).
> Tables that the post shows only as images were transcribed from those images (Figures 16, 24, 28, 29, 30, 37, 38).

## Introduction

Raschka says Jev (TypeSafe AI, https://typesafe.ai/blog/introducing-system-one-models-and-jev) became "quite a cultural
phenomenon" in the two weeks before the post. His view moved from "classifiers used to be my bread & butter; I can easily
build this myself" to "wow, this actually works better than I thought." Framing: frontier LLMs can do the same
classification but slower and dearer; a task-specific classifier will usually beat Jev on one narrow task; Jev's selling
point is generality. He discloses no affiliation, no free access, and calls the article not an endorsement. The Jev
methodology section is labelled "an educated guess."

- [Figure 1](https://substack-post-media.s3.amazonaws.com/public/images/a5481509-dd15-42bf-9ce0-29a292276016_6461x3353.png): quick overview of the Jev API.

## 1. Language modeling and classification in the pre-transformer era

### 1.1 Bag-of-words: naive Bayes, logistic regression, and XGBoost

- Bag-of-words (BoW) turns variable-length text into a fixed-length count vector (one slot per vocabulary word, mostly
  zeros; TF-IDF as a normalisation) so classic classifiers (naive Bayes, logistic regression, SVMs, random forests, XGBoost)
  can use it. Uses: news topic, spam (Gmail's early filter "allegedly" naive Bayes + BoW). Naive Bayes text classification
  dates to at least 1961 (Maron, *Automatic Indexing: An Experimental Inquiry*). His own 2014 arXiv tutorial: https://arxiv.org/abs/1410.5329.
- Weakness: loses word order ("the dog bites the man" = "the man bites the dog"); n-grams partly recover it at the cost of
  vocabulary size.
- Opinion: BoW + logistic regression "remains my go-to baseline for every text classification problem."
- Example data: IMDb movie reviews (https://ai.stanford.edu/~amaas/data/sentiment/), balanced, 25,000 test reviews.
- [Figure 2](https://substack-post-media.s3.amazonaws.com/public/images/7900489a-1e4c-4997-974b-3b5f89d62465_1050x1318.png): his 2014 naive Bayes / BoW tutorial.
- [Figure 3](https://substack-post-media.s3.amazonaws.com/public/images/5f041db0-58bb-43db-a07c-ac672b613470_7769x3701.png): a bag-of-words representation.
- [Figure 4](https://substack-post-media.s3.amazonaws.com/public/images/1fbb03a2-bb84-407b-8e08-f298e72a85f2_3907x2469.png): logistic regression tutorial (https://github.com/rasbt/machine-learning-book/blob/main/ch08/ch08.ipynb); **89.9%** IMDb accuracy.

### 1.2 Deep neural networks for text classification

**1.2.1 Word embeddings.** Dense per-word vectors (Word2Vec, GloVe, or a trained embedding layer); context-independent at
lookup ("bank" is one vector). Links to his LLMs-from-scratch ch. 2 notebooks.
- [Figure 5](https://substack-post-media.s3.amazonaws.com/public/images/b7384f46-af12-44c5-a1b4-5ec0c023094c_7579x3181.png): creating word embeddings.

**1.2.2 RNNs.** Read one token at a time, carry a fixed-size hidden state; LSTM (1997), GRU (2014), xLSTM (2024); state-space
models share the fixed-state bottleneck. Attention first appeared in RNNs. A from-scratch LSTM gets **85.66%** on IMDb (below
BoW + LR at 89.9%), with clear overfitting; pre-training then fine-tuning (ULMFiT, https://arxiv.org/abs/1801.06146, 2018)
reached **95.4%**.
- [Figure 6](https://substack-post-media.s3.amazonaws.com/public/images/63c0da28-827d-4a59-9fed-cb283e57ad1b_7776x4745.png): an RNN classifier (unrolled).
- [Figure 7](https://substack-post-media.s3.amazonaws.com/public/images/dc3e8a8d-8e3b-4989-8b42-564cf84c80b6_7675x3257.png): rolled vs unrolled RNN.
- [Figure 8](https://substack-post-media.s3.amazonaws.com/public/images/dbb8bd45-04a6-47c5-bbdb-b87fe171b32b_3866x2127.png): LSTM tutorial (https://github.com/rasbt/machine-learning-book/blob/main/ch15/ch15_part2.ipynb), 85.66%; training accuracy higher (overfitting).
- [Figure 9](https://substack-post-media.s3.amazonaws.com/public/images/5e8aeef2-95e4-415a-8f5f-93895d1f2eb2_4470x2477.png): annotated ULMFiT figure.

**1.2.3 CNNs.** Learned filters slide over windows of word embeddings (window 3 over "the movie had surprisingly good
acting" gives four windows); global max-pooling makes the output length-independent; positions compute in parallel. His
text CNN: **90.07%** on IMDb (architecture-dependent; cf. AlexNet ~62.5% vs ConvNeXt V2-H 88.9% ImageNet top-1).
- [Figure 10](https://substack-post-media.s3.amazonaws.com/public/images/bacd70f8-4159-41b2-a812-9ab65de29f2b_7288x2635.png): a CNN for images.
- [Figure 11](https://substack-post-media.s3.amazonaws.com/public/images/5b710ad1-6fad-47a7-9b79-192b61097194_7935x7279.png): text CNN step by step, one filter.
- [Figure 12](https://substack-post-media.s3.amazonaws.com/public/images/88ad7444-1759-4966-a53e-4744f13ff889_7748x2519.png): text CNN with several filters.
- [Figure 13](https://substack-post-media.s3.amazonaws.com/public/images/91765c78-a4d8-48bf-ade0-10b515441b8a_4442x2189.png): IMDb CNN, 90.07% (code: https://github.com/rasbt/deeplearning-models/blob/master/pytorch_ipynb/cnn-nlp/cnn_imdb.ipynb).

## 2. Transformers

Original transformer (https://arxiv.org/abs/1706.03762, 2017): encoder-decoder for translation, easily adapted to
classification.
- [Figure 14](https://substack-post-media.s3.amazonaws.com/public/images/d94764e4-62ed-4c95-b468-80fb6f9a6621_6217x7963.png): the original architecture.

### 2.1 Encoder-style language models

Early years were BERT (Google, encoder) vs GPT (OpenAI, decoder). Encoders were "natural text classifiers" via the `[CLS]`
token (BERT, https://arxiv.org/abs/1810.04805); GPT did zero/few-shot classification as an emergent ability. Both are
pre-trained then fine-tuned. ModernBERT (https://arxiv.org/abs/2412.13663, 2024) is "often my go-to for classification";
~95% on IMDb with little tuning.
- [Figure 15](https://substack-post-media.s3.amazonaws.com/public/images/59456abf-cf68-4d6e-9c92-73ad3e593665_4982x3213.png): annotated BERT figure.
- [Figure 16](https://substack-post-media.s3.amazonaws.com/public/images/f4cc75fe-608d-47fc-95aa-14f3f2b58b83_4147x2928.png): models fine-tuned on IMDb (code: https://github.com/rasbt/LLMs-from-scratch/tree/main/ch06/03_bonus_imdb-classification); little tuning, "could be improved by 1-2%". Transcribed:

| # | Model | IMDb test accuracy |
|---|---|---:|
| 1.1 | 124M GPT-2 baseline | 91.88% |
| 1.2 | 124M GPT-2 baseline (with Muon) | 92.40% |
| 2 | 340M BERT | 90.89% |
| 3 | 66M DistilBERT | 91.40% |
| 4 | 355M RoBERTa | 92.95% |
| 5 | 304M DeBERTa-v3 | 94.69% |
| 6 | 149M ModernBERT Base | 93.79% |
| 7 | 395M ModernBERT Large | 95.07% |
| 8 | Logistic regression baseline | 88.85% |

### 2.2 Decoder-style LLMs

You can prompt an LLM to classify, but for structured output in a fixed domain that is "unnecessarily brittle and
inefficient". Instead replace the vocabulary output layer with a small classification head (his book, ch. 6). With causal
attention the token used for the decision must be one that has seen the whole sequence (the last token). Advantage over
BERT variants: many modern open-weight backbones (Qwen3 0.6B up to Kimi, GLM, DeepSeek), though ">1B ... could be a bit
overkill". GPT-2 124M: ~92% on IMDb. Links his "Building an AI Text Detector From Scratch" article.
- [Figure 17](https://substack-post-media.s3.amazonaws.com/public/images/0be7e4aa-5436-46cf-8e85-809f776f39e1_4920x3189.png): prompting an LLM to classify a review.
- [Figure 18](https://substack-post-media.s3.amazonaws.com/public/images/9065aa90-305d-4373-bf45-73d82ec86768_1943x1931.png): swapping a GPT's output layer for a classification head.
- [Figure 19](https://substack-post-media.s3.amazonaws.com/public/images/c6bc5509-085e-4b93-b557-566a5a05f0ff_2940x2021.png): BERT vs GPT attention masks.
- [Figure 20](https://substack-post-media.s3.amazonaws.com/public/images/b41880c4-c2e4-4deb-b97b-13d82b47c8c7_4164x2928.png): GPT-2 124M ~92% on IMDb.

### 2.3 Encoder-decoder style architectures

T5 (https://arxiv.org/abs/1910.10683, 2019): span-corruption pre-training. Two ways to classify: a classification head, or
"text-to-text classification" (the decoder emits the label word). GPT-style LLMs do text-to-text out of the box; T5 is
usually fine-tuned for it. Mentions DeepSeek V4.1 Flash as a recent (causal-encoder) encoder-decoder.
- [Figure 21](https://substack-post-media.s3.amazonaws.com/public/images/b15fdb01-219a-4373-8b17-a15a9956436a_6256x4375.png): T5's changes to the original transformer.
- Figure 22: the Figure 17 image reused (text-to-text classification with a GPT model).
- [Figure 23](https://substack-post-media.s3.amazonaws.com/public/images/e2a84f80-0975-4ddc-aee6-d3579f50edc5_5046x1890.png): classification head vs text-to-text.

## 3. Jev overview

### 3.1 Jev vs existing text-to-text classification

Jev is proprietary (TypeSafe AI, out of stealth weeks earlier), cheap, and claims to match GPT-5.6 Luna on decisions at
orders of magnitude lower cost. Raschka treats Luna as the "text-to-text" approach. His reasons for the hype: a nice API and
good results across tasks without fine-tuning: "the ChatGPT moment for classification". He shows two videos (support-ticket
routing; playing Tetris in real time via the Choice API). Training is undisclosed except a founder's phrase
"Reinforcement Learning for Calibrated Decisions".

- [Figure 24](https://substack-post-media.s3.amazonaws.com/public/images/13d2d71d-7704-433d-9d4d-ac8a87ef3b8a_5584x3084.png): TypeSafe's chart "Average of 4 workflows: accuracy vs cost" (vendor benchmark). Read off the chart (approximate): Jev ~68% at ~$0.0004 per workflow; GPT-5.6 Luna workflow ~67% at ~$0.003 (Luna prompt ~52% at ~$0.008); terra ~68% (workflow) at ~$0.03; sol ~74% at ~$0.08; Opus 5 ~73% (workflow) at ~$0.18; Sonnet 5 ~68%; DeepSeek v4 flash ~64%, v4 pro ~66%; Haiku 4.5 ~54% (one prompt point at 18%). Raschka's annotation: "Jev is on GPT-5.6 Luna's level while being much cheaper."

### 3.2 The Jev API

Three question types:
- [Figure 25](https://substack-post-media.s3.amazonaws.com/public/images/4ac35084-507b-4f08-856c-dd449f43ebf2_6461x3353.png): **Choice**, multi-class.
- [Figure 26](https://substack-post-media.s3.amazonaws.com/public/images/b63ec307-8be5-48c5-a0f3-21dbe461c78d_6461x3353.png): **Noul**, a "yes" probability; binary, or multi-label via several questions.
- [Figure 27](https://substack-post-media.s3.amazonaws.com/public/images/b3eb167e-451f-4bb8-8c8d-4da0a781ab5d_6461x3354.png): **Score**, ordinal against a rubric (levels 0, 1, 2).
- [Figure 28](https://substack-post-media.s3.amazonaws.com/public/images/e962a588-c3d3-453a-9e1a-abfccfd4e7b7_4455x1747.png): API cheat sheet, transcribed:

| Jev decision type | How we could implement it (Raschka) |
|---|---|
| `noul`, yes/no | Score "yes" and "no", return P(yes). Binary and multi-label (several binary questions). |
| `choice`, user-defined options | Score each candidate, softmax to sum to 1, return the option and the distribution. Multi-class. |
| `score`, rating against a rubric | Score each rubric level, softmax, return the probability-weighted mean level. Ordinal. |

### 3.3 Jev classifying IMDb

Choice request (as in the post):

```
curl -sS https://api.typesafe.ai/v1/systemone \
  -H "Authorization: Bearer $TYPESAFE_API_KEY" -H "Content-Type: application/json" \
  -d '{"model": "jev-1.13.0",
       "state": "The acting was excellent and the story kept me engaged throughout. I would happily watch this movie again.",
       "questions": {"sentiment": {"type": "choice",
         "instructions": "What is the overall sentiment of this movie review?",
         "criteria": {"negative": "An overall unfavorable opinion of the movie",
                      "positive": "An overall favorable opinion of the movie"}}}}'
```

Response: `choice: positive`, `confidence: 1.0`, `probabilities: {negative: 0.0, positive: 1.0}`, usage 342 input / 32
output tokens. He notes `confidence` summarises how concentrated the distribution is and differs from the winning class's
probability; Jev's docs say the probabilities are well calibrated. The Noul form (`is_positive`, criteria for true/false)
returned `noul: 0.98` on the same text (328 / 21 tokens), "interestingly" not 1.0. Noul probabilities across several
per-class questions need not sum to 1.

Full IMDb test set (25,000 reviews):

| API | Accuracy | Correct | Runtime | Input tokens | Cost |
|---|---:|---:|---|---:|---:|
| Choice | 96.47% | 24,117 | 22 min 24 s | 15,456,663 | $0.6492 |
| Noul | 96.20% | 24,050 | 23 min 3 s | 15,106,663 | $0.6345 |

The Choice/Noul gap may be noise: runs are not deterministic.
- [Figure 29](https://substack-post-media.s3.amazonaws.com/public/images/477155d8-c954-471a-aa18-f456ad16b221_4445x682.png): repeated runs, transcribed:

| Run | Correct | Accuracy |
|---|---:|---:|
| Original Choice | 24,117 | 96.468% |
| Repeated Choice | 24,113 | 96.452% |

He attributes non-determinism to batch-dependent GPU kernels (Horace He, Thinking Machines,
https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/). Caveat in the post: unknown whether IMDb test
was in Jev's training data.

- [Figure 30](https://substack-post-media.s3.amazonaws.com/public/images/9933df13-7e9b-4144-86f3-b53781791e85_4776x2169.png): ModernBERT vs Jev (ModernBERT runs on a DGX Spark; "1-2%" more possible with tuning). Transcribed:

| # | Model | Fine-tuning time | Test evaluation time | Test accuracy |
|---|---|---|---|---:|
| 1 | ModernBERT (149M), 256-token window | 23m 03.6s | 7m 25s | 93.908% |
| 2 | ModernBERT (395M), 256-token window | 1h 00m 47.2s | 19m 48s | 95.212% |
| 3 | ModernBERT (149M), no token cut-off | 1h 14m 28.0s | 21m 47s | 95.456% |
| 4 | ModernBERT (395M), no token cut-off | 3h 01m 30.3s | 55m 41s | 96.504% |
| 5 | Jev | - | 23m 3s | 96.468% |

Rule of thumb he gives: a cheap LLM for one-off decisions, a fine-tuned classifier for repeated ones; Jev-likes now cover
both, except very high-volume narrow tasks, where fine-tuning still pays.

## 4. BERT- and GPT-style models with a Jev API

He built a Jev-like ModernBERT right after the release and chose not to publish it: putting the API on a model and
fine-tuning on a few tasks is "trivial", making it work across very different tasks (e.g. Tetris) is not.

Recipe for a Choice API on any BERT/GPT/T5 backbone:
- A classification head with **one output node** instead of one per class.
- For each candidate option, feed input text + task instruction + that candidate's description; the head maps the pooled
  representation `h_i` to a scalar `s_i = w^T h_i + b` (a logistic-regression layer, shared across candidates).
- Softmax over the N scores gives N probabilities; the head's parameters do not depend on N, so the option set can change
  without changing the architecture.
- `h_i`: the `[CLS]` final hidden state (optionally pooled) for BERT-style; the last non-padding token for GPT-style.
- Train backbone and head jointly with cross-entropy on the correct option. Generalisation to unfamiliar tasks "depends on
  the training data."
- Why transformers over RNN/CNN for this: they scale with pre-training data and use context through attention (argument,
  not measured here).
- [Figure 31](https://substack-post-media.s3.amazonaws.com/public/images/dd09a503-5698-4cc1-8692-7a9bcb81ffba_1943x1931.png): GPT with a one-node output head.
- [Figure 32](https://substack-post-media.s3.amazonaws.com/public/images/4a0ef4b0-2afd-4ba9-9be5-95440b50b2aa_6723x3497.png): BERT/GPT with a Jev-like Choice API (one pass per option, 3-class ticket example: billing / technical / account).

## 5. Jev architecture and training algorithm

Undisclosed. His guess: "something small similar to ModernBERT, hence, the low latency" (explicitly a guess). Data: the
TypeSafe CEO said on X that 100% of their data is synthetic, "but not the type of crap that is just spit out from an LLM".
Raschka thinks most effort went into data curation. Anecdote (~8 years ago, social-science collaboration): days of tuning
BoW and BERT gave ~2-5%; doubling ~300 hand-labelled examples gave a ">10-20%" boost. Advice: plot learning curves.
- [Figure 33](https://substack-post-media.s3.amazonaws.com/public/images/cfa6e6db-1008-4418-b319-41f1bba35410_6020x2962.png): more data often helps more than more hyperparameter tuning (illustrative).

TypeSafe's blog names the training method Reinforcement Learning for Calibrated Decisions (RLCD), with "a new model
architecture, parallel sampler". RLCD is unpublished; the nearest published method is RLCR (*Beyond Binary Rewards:
Training LMs to Reason About Their Uncertainty*, https://arxiv.org/abs/2507.16806, 2025). No established link.

### 5.1 On calibration

Calibration adjusts probabilities to match observed frequencies on held-out data (among reviews given ~74% positive, ~74%
should be positive); recommended for any production model whose probabilities are used. Pointer: scikit-learn's
calibration docs. His method in the AI-detector project: **temperature scaling**: divide logits by a learned T > 0 fitted
by minimising CE on a calibration set with weights frozen; T > 1 softens, 0 < T < 1 sharpens; argmax unchanged.
- [Figure 34](https://substack-post-media.s3.amazonaws.com/public/images/642f91a5-0550-4d6d-8f70-85a1b77330c7_5046x2578.png): how calibration works.

### 5.2 RLCR

RLVR rewards 1/0 for correctness. RLCR has the model generate reasoning, answer, an uncertainty analysis and a confidence
q in tags, with reward **R = c - (q - c)^2** (c = 1 if correct): wrong at q = 0.9 gives -0.81, wrong at 0.2 gives -0.04,
right at 0.9 gives 0.99. The squared term is the Brier penalty on the stated probability of being correct. The paper uses
Qwen2.5-7B.
- [Figure 35](https://substack-post-media.s3.amazonaws.com/public/images/23d71def-2f99-4f25-a66e-9ee6e656c6d8_6070x2504.png): RLCR overview.
- [Figure 36](https://substack-post-media.s3.amazonaws.com/public/images/a0028832-3d38-4d48-93f4-fc4547db9fd0_6661x3645.png): RLCR results (paper's Table 1(a)).

Numbers he quotes from the paper: HotpotQA ECE 0.37 -> 0.03 vs RLVR at similar accuracy (62.1% vs 63.0%); average over six
other datasets ECE 0.46 -> 0.21 and accuracy 53.9% -> 56.2%. "RLVR + Classifier" in the figure is a separate binary
correctness classifier.

For a Jev-like model he proposes two adaptations (his ideas, not Jev's known method): calibration rewards applied directly
to typed decisions from a classification head (no reasoning text), or, by analogy with DPO replacing RLHF with a supervised
loss, simply minimise **CE + Brier** on the classifier's probabilities. He did the latter on ModernBERT:

- [Figure 37](https://substack-post-media.s3.amazonaws.com/public/images/420c9874-1fe8-4a87-9c7d-483966e77879_6239x2423.png): ModernBERT with RLCR-inspired Brier loss; mean of nine datasets weighted equally (ECE over 15 bins), plus IMDb. Transcribed:

| Metric | CE | CE + temperature | CE+Brier | CE+Brier + temperature |
|---|---:|---:|---:|---:|
| Macro accuracy (9 sets) ↑ | 73.20% | 73.20% | 73.27% | 73.27% |
| Negative log-likelihood ↓ | 1.1622 | 0.7706 | 1.1434 | 0.7691 |
| Brier score ↓ | 0.4093 | 0.3574 | 0.4074 | 0.3569 |
| Expected calibration error ↓ | 15.20% | 5.63% | 14.94% | 5.48% |
| IMDb accuracy ↑ | 94.85% | 94.85% | 94.84% | 94.84% |

Averaged, calibration helped; ECE got slightly worse on one of the nine sets. (The nine datasets are not named in the post.)

### 5.3 Is calibration necessary?

At the population optimum, CE (and Brier, also strictly proper: Gneiting and Raftery 2007) already yields calibrated
probabilities. With finite data a network keeps lowering training CE by growing confident, so test probabilities degrade
even as accuracy improves (Guo et al. 2017, *On Calibration of Modern Neural Networks*: networks "can overfit to NLL
without overfitting to the 0/1 loss"). Adding Brier reweights errors; whether it helps must be checked on held-out data;
in his runs it gave "only a very small additional calibration benefit". The Brier term matters more in RL, where no CE is
being minimised.

## 6. Who is Jev for?

People who want to skip fine-tuning a custom classifier per task and to avoid paying for frontier LLMs.

### 6.1 Use cases

Email sorting, spam, prioritisation; inside agent harnesses as a prompt-injection pre-screener, reasoning-effort selector,
skill selector, judge for evaluation or self-refinement, and file finder for context. Cites an arXiv survey of 2,170
Jev-related projects (https://arxiv.org/abs/2609.30216).

### 6.2 Jev clones and local Jevs

Hundreds to thousands of clones; most he looked at were ModernBERT or Qwen with SFT and a Jev-like API; none matched Jev's
breadth in his view ("like comparing Alpaca ... to GPT-6"). GLiNER (https://github.com/urchade/GLiNER), ~3 years old and not
the same thing, can do similar jobs; a third-party benchmark shows Jev stronger:

- [Figure 38](https://substack-post-media.s3.amazonaws.com/public/images/6009d1b8-e300-41b8-bf62-3eb10532025f_4065x1743.png): GLiNER vs Jev, from https://github.com/AbdelStark/jev-benchmarks/blob/main/results/reports/btzsc-pilot-v1.md. Transcribed:

| Dataset | Model | Accuracy | Macro-F1 | Brier ↓ | NLL ↓ | ECE ↓ | Coverage at ≤5% empirical error | p50 latency |
|---|---|---:|---:|---:|---:|---:|---:|---|
| AG News | GLiNER2.5 | 0.700 | 0.659 | 0.413 | 0.742 | 0.124 | 0.240 | 44.9 ms local CPU |
| AG News | Jev | **0.910** | **0.905** | **0.146** | **0.495** | **0.064** | **0.830** | 255.9 ms hosted |
| Banking77/BTZSC | GLiNER2.5 | 0.610 | 0.569 | 0.521 | 1.444 | 0.062 | 0.270 | 295.5 ms local CPU |
| Banking77/BTZSC | Jev | **0.870** | **0.857** | **0.179** | **1.064** | **0.054** | **0.860** | **246.4 ms hosted** |
| DAIR Emotion | GLiNER2.5 | 0.440 | 0.407 | **0.668** | **1.381** | **0.117** | **0.020** | **43.3 ms local CPU** |
| DAIR Emotion | Jev | 0.480 | 0.479 | 0.846 | 5.588 | 0.351 | 0.000 | 236.3 ms hosted |

He wants an open-weight Jev mainly for privacy rather than cost, and expects it to take time.

**Update 1 (29 Sept, 10:20 PT):** OpenAI announced a Decisions API at DevDay 2026, described as focusing Luna on
user-defined questions with finite answers, text or image context, limited preview
(https://openai.com/index/devday-2026-recap/).
- [Figure 39](https://substack-post-media.s3.amazonaws.com/public/images/fecf4d0d-2865-4ad1-8322-24f27fc3d323_1840x876.png): OpenAI's Decision API announcement.

**Update 2:** Contrastive Language Models (https://contrastive-lm.notion.site/): IMDb **82.90%** (vs Jev 96.47%) and fails
his Tetris test. Laya (https://huggingface.co/convaiinnovations/laya): IMDb **92.33%**, fails Tetris worse. A reader reported
Jev is weak on non-English legal review; he suggests a cheap translation model in front.

## Conclusion

Jev is not fundamentally new ("just a classifier" with a nice API) but works surprisingly well across many tasks, the
"plug-and-play version of dedicated classifiers". The bar for fine-tuning a specialist is now higher. He does not expect
Jev-likes to unlock new capabilities, but sees them making agent harnesses faster and cheaper by handling decisions for
large models.
