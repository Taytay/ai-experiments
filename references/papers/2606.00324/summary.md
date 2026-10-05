# LLMs Need Encoders for Semantic IDs Too

- arXiv 2606.00324 (v1, 29 May 2026) - https://arxiv.org/abs/2606.00324
- Xiangyi Chen, Zelun Wang, Xinyi Li, Yi-Ping Hsu, Jaewon Yang, Jiajing Xu (Pinterest)
- Venue: preprint (ACM template with placeholder conference fields)
- Source: references/papers/2606.00324/paper_flat.tex

## One-paragraph summary
In generative recommendation, items are hierarchical Semantic IDs (SIDs) made by RQ-VAE: here 5 levels of 2048 codes, built from PinCLIP multimodal embeddings. They are added to an LLM's vocabulary as L x K flat tokens. A code's meaning depends on the codes before it (code 505 after 1273 is sneakers; after 974 it is vintage art). A flat token embedding cannot express this, and at level l there are K^(l-1) possible contexts. PrefixMem is a "SID encoder" in the sense of a vision encoder in a multimodal LLM. At each SID position it hashes the prefix n-grams (orders 1 to 4, 4 hash heads, multiply-XOR with primes, mod a table size T) into embedding tables, sums them, projects with a near-zero-initialised W_out, and adds the result to the token's input embedding. On Qwen3-1.7B with matched compute, teacher-forced level-5 accuracy goes from 37.6 to 54.8 with 2M rows, and to 64.0 with classification-pretrained 5M tables. Full-SID Recall@100 goes from 9.5 to 11.8. Gains concentrate on "unreachable" examples (+77% relative) and rare prefixes (+115%). Qwen3-0.6B with PrefixMem (54.1) beats Qwen3-4B without it (45.0). A 336M-parameter 4-layer transformer encoder in the same slot barely helps (39.6). The authors conclude the problem is memory capacity, not computation.

## Problem
Current SID-LLMs (OneRec, LC-Rec, PLUM-like) give each code one embedding, whatever its prefix. The LLM must then learn, inside its shared weights, which prefix-code combinations exist and are popular, across a combinatorial, sparse and skewed space. Middle levels also suffer the "hourglass" collapse. Smart initialisation of new token embeddings does not address prefix dependence. Constrained decoding prevents invalid SIDs but does not teach the hierarchy. Massive continual pretraining (PLUM) works but costs hundreds of billions of tokens and ties SID knowledge to one checkpoint.

## Method
- How SIDs are built (Section 4.1): each item's PinCLIP multimodal embedding is quantised by RQ-VAE into a 5-level SID with 2048 codes per level. The codebook is trained over the full billion-item catalog. Several items can share a SID (collisions). Text for each item is a VLM-generated image description (Table 1 examples: "1273, 505, 1934, 1882, 1288" is a beige and brown Nike sneaker).
- LLM adaptation (Section 3.1): the tokenizer gets L x K new SID tokens. Causal LM loss is computed on three sequence types sampled at random in one stage:
  1. interleaved (text_1, sid_1, text_2, sid_2, ...) or SID-first;
  2. single pair (text_i, sid_i) or (sid_i, text_i);
  3. SID-only sequences (sid_1, sid_2, ...).
- PrefixMem (Section 3.2, Fig. 2, Eqs. 1-2): for the prefix (c_1..c_l), for n-gram order n in 1..min(l, N_max) and head h in 1..H, idx_{n,h} = (XOR_{i=1..n} c_i * p_{i,h}) mod T. Tables E_{n,h} have shape T x d/H. Vectors are concatenated across heads, summed across orders, and projected: m_l = W_out * sum_n concat_h E_{n,h}[idx]. m_l is added to the input embedding of c_l before the transformer, so the LLM sees a prefix-specific representation when predicting c_{l+1}. W_out starts near zero.
- Defaults (Section 4.2): H=4, T=2M, d_mem=256, N_max=4. Active only at levels 4 and 5. Encoder learning rate is 5x the LLM's. About 2B table parameters.
- Encoder pretraining ladder (Section 3.3): (1) a classification head predicting c_{l+1} from m_l with cross-entropy (transition statistics only, no transformer FLOPs); (2) training jointly with a generative retrieval model (Tiger), which captures behavioural patterns; (3) training jointly with a small LLM (Qwen3-0.6B), which grounds the tables in language. Tables are then loaded with a fresh W_out.
- Comparator: SID-Transformer, a 4-layer causal transformer (336M parameters) over per-level prefix embeddings, injected at the same point.
- Data: about 10M subsequences of up to 32 engagements (about 240M item occurrences, tens of millions of unique items). Train and eval are split by time and by disjoint users. 100K eval examples, each predicting the SID at a random position from the preceding history.
- Models: Qwen3 0.6B, 1.7B (default) and 4B, Llama 3.2 1B, Gemma 3 1B. 50K steps, learning rate from a sweep on 1.7B and reused for all.
- Metrics:
  - TF-L_l: teacher-forced per-level top-1 accuracy (primary)
  - Full-SID Recall@K by beam search
  - BLEU for SID-to-text
  - SID hit rate (whether the predicted prefix bucket is non-empty)
  - reachable/unreachable split (whether the ground truth is in the top 10 at every level L1-L4)

## Experiments and results
- Table 2 (Qwen3-1.7B, TF-L4 / TF-L5): baseline 33.3 / 37.6; 500K rows 40.1 / 49.7; 2M rows 42.6 / 54.8 (+28% / +46% relative); 5M rows 43.4 / 57.2. Levels 1-3 are unchanged (about 42 / 29 / 29). When also active at L3: L3 goes 29.2 to 30.9 (+6%), L4 34.5 to 43.9, L5 37.6 to 54.8. The gain grows with prefix length.
- Fig. 3: the gap opens within a few thousand steps and keeps widening. Baseline plateaus near 35, the 2M encoder reaches 53, the pretrained encoder 58.
- Table 3, initialisation (TF-L4, TF-L5, R@100, BLEU):
  - random 2M: 42.6, 54.8, 11.3, 29.1
  - random 5M: 43.4, 57.2, 11.6, 32.3
  - classification-pretrained 2M: 43.2, 58.5, 11.2, 26.3
  - classification-pretrained 5M: 49.6, 64.0, 11.8, 27.2 (+70% relative TF-L5)
  - LLM-pretrained (1.7B): 45.3, 57.0, 11.9, 33.1
  - LLM-pretrained (0.6B, moved to 1.7B): 45.3, 56.8, 11.8, 33.1 (the tables transfer across model sizes)
  - Tiger-pretrained 2M: 46.7, 61.2, 12.3 (best R@100), 28.2
  - Each pretraining objective biases a different metric.
- Table 4, Recall@20/30/50/100: baseline 6.0 / 6.9 / 8.0 / 9.5; 5M 6.7 / 7.8 / 9.4 / 11.6, a relative gain growing from +11% to +22% with beam width. Table 5: unique L1 clusters in the top-100 beam rise from 5.5 to 6.2.
- Table 6, SID hit rate at L5: baseline 50.7%; 5M 58.9; classification-pretrained 5M 65.1. The baseline hallucinates non-existent SIDs for about half of predictions.
- Table 7, TF-L5 reachable / unreachable: baseline 43.3 / 36.4; pretrained 5M 44.6 / 64.5 (+3.0% vs +77.2% relative).
- Table 8, by L4 prefix popularity (rare / medium / popular): baseline 26.8 / 34.8 / 52.5; classification-pretrained 5M 57.6 / 75.6 / 70.4. Rare prefixes gain +115%, popular +34%.
- Table 9, across models (TF-L5, R@100, without and with the 2M encoder):
  - Qwen3 0.6B: 32.1, 8.4 to 54.1, 10.5
  - Qwen3 1.7B: 37.6, 9.5 to 54.8, 11.3
  - Qwen3 4B: 45.0, 10.9 to 55.8, 12.0
  - Llama 3.2 1B: 39.2 to 54.5
  - Gemma 3 1B: 37.3 to 54.4
  - Relative gain shrinks with model size (+69%, +46%, +24%).
- Table 10, Tiger (35M, non-LLM): TF-L4 27.6 to 46.3, TF-L5 27.4 to 60.6 (+121%), R@10 1.01 to 1.35.
- Table 11, architecture (TF-L5, BLEU): baseline 37.6, 23.3; PrefixMem 54.8, 30.9; SID-Transformer (4-layer, 336M) 39.6, 23.4.
- Table 12, design ablation: multi-scale prefix vs multi-scale suffix vs single full-prefix hash give TF-L5 54.8 / 54.6 / 54.0, so the design is robust.
- Overhead: under 0.02% FLOPs (about 2M per 5-level span vs about 17B for the LLM), under 3% training throughput loss, about 10 GB extra peak GPU memory (tables plus AdamW state). Serving cost is one lookup and one projection per SID token.

## Limitations
- Evaluation leans on teacher-forced per-level accuracy. End-to-end recall gains are much smaller (9.5 to 11.8 at R@100, under 1 point at R@20) and absolute recall is low.
- Internal inconsistencies: the random-init 2M BLEU is 29.1 in Table 3 but 30.9 in Tables 11-12. Baseline TF-L4 is 33.3 in Table 2 but 34.5 in the L3-L5 run. In Table 7 the baseline does better on unreachable than reachable examples only at L5, which is odd and not discussed.
- About 2B extra table parameters and 10 GB of optimizer state. The hash table must scale with the number of distinct prefixes.
- By the authors' own statement it helps only with 3 or more levels and large prefix spaces. Level 2 (2048 contexts) is memorised by the LLM directly. It also depends on the RQ-VAE transitions having structure.
- One dataset (Pinterest), one SID construction (RQ-VAE on PinCLIP). No comparison with OneRec-style RL or constrained decoding, which they call complementary. No measurement of language-capability loss (only argued).

## Relevance to this workspace
- Row 197 (semantic IDs for categories): the paper is a warning about scale. PrefixMem's case rests on a combinatorial prefix space (2048^3 to 2048^4 contexts, tens of millions of items, rare prefixes). We have at most about 100 categories per user and 64 cross-household clusters. Any category SID is 2 levels at most (cluster then sub-cluster) with a tiny code space. By the paper's own limitation ("for SID hierarchies with fewer than 3 levels, the encoder would provide little benefit"), the LLM can memorise such codes as flat tokens, so PrefixMem itself does not transfer to category SIDs.
- What does carry over to row 197:
  1. How SIDs are built: RQ-VAE over a content embedding, with the codebook trained over the whole catalog. For us that is the bge or decider embedding of category name plus filings, pooled across households, which the 64 clusters already approximate as level 1.
  2. The three-way training mix (interleaved text and SID, single pairs in both directions, SID-only sequences) as the recipe for grounding new tokens. If categories get codes in the decider, train name-to-code and code-to-name pairs alongside history sequences.
  3. The point that a flat new token learns its meaning slowly from scratch (Fig. 3). Initialise codes from cluster centroids rather than at random.
- The deeper finding is about memory versus computation, and it does transfer. A 336M transformer could not memorise a sparse combinatorial mapping that an O(1) hash lookup stored easily, and a 0.6B model with tables beat a 4B model without them. Our analogue is the per-household payee-to-category mapping, which is pure memorisation. It explains why non-parametric history lookups (kNN 68.4, MaxSim 69.6) come close to the 4B decider (73.6), and suggests giving the decider explicit memory rather than more parameters. Two options:
  - (a) Inject a retrieved history vector at the answer-slot or category tokens (additive, near-zero-initialised projection, as PrefixMem does), instead of only listing history rows as text.
  - (b) Use a hashed n-gram memory keyed on (household-hash, normalised payee tokens) or (payee n-grams), trained to predict the category cluster. It would be cheap to pretrain with a classification head, which is the paper's best pretraining for top-1.
- Evaluation ideas worth copying:
  - Split accuracy by "reachable" (was the right category in the top-k under some prior) and by prefix popularity. For us that means splitting by payee frequency in the user's history (first-time, rare, frequent), where the paper saw +115% on rare.
  - The SID hit-rate idea is irrelevant for us because we score a closed category list (no hallucination possible).
- Licences: the Qwen3, Llama and Gemma bases used are not all open-licence by our rule. Only the idea is relevant, and it is base-agnostic (it worked across families).
