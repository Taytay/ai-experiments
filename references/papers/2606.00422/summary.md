# UniPinRec: Unifying Generative Retrieval and Ranking at Pinterest Scale

- arXiv 2606.00422 (v1, 29 May 2026) - https://arxiv.org/abs/2606.00422
- Hanyu Li, Yi-Ping Hsu, Aditya Mantha, Prabhat Agarwal, Laksh Bhasin, Jialu Wang, Hongtao Lin, Bella Huang, Yaxin Li, Xinyi Li, Chuxi Wang, Kousik Rajesh, Hooshmand Shokri Razaghi, Shunyao Li, Zongyue Qin, Jaewon Yang, James Li, Dhruvil Deven Badani, Jiajing Xu, Charles Rosenberg (Pinterest)
- Venue: RecSys '26 (Proceedings of the 20th ACM Conference on Recommender Systems), DOI 10.1145/3773078.3831921
- Source: references/papers/2606.00422/paper_flat.tex

## One-paragraph summary
UniPinRec extends PinRec, Pinterest's production retrieval model (a causal transformer over the user's engagement history that autoregressively generates query embeddings for a Faiss ANN index), into one model that also ranks. One 12-layer backbone reads one non-interleaved user sequence. Retrieval uses next-item sampled softmax at past positions. Ranking uses per-action binary heads on candidate tokens appended after the history, where each candidate attends to the history but not to the other candidates. The three main ideas are: (1) Masked Action Modeling (MAM): each history item's action multi-hot is concatenated to its embedding along the feature axis and randomly replaced by a [MASK] class, so ranking supervision needs no interleaved action tokens and the sequence does not get longer; (2) training examples that pair a past action sequence with a later impression slate (a "feedview"), trained with one joint loss in a single stage; (3) serving where the ranking process reads the retrieval process's history KV cache through shared GPU memory (CUDA IPC), which turns ranking into an O(nk) decode step. Offline, ranking Hit@3 is 0.10096 against 0.08801 for the production TransActV2+DCNv2 ranker (+14.7%), and retrieval Recall@10 is 0.77659 against PinRec's 0.77486 (Table 1). Online, deployed as L0 retrieval plus L1 ranking, it gave +0.95% Board More Ideas saves and +0.91% push opens, with -11.1% end-to-end latency and +63.6% QPS against serving the two separately (Tables 3, 5, 6). **The paper does not use semantic IDs.** "Generative retrieval" here means autoregressive generation of dense query embeddings plus ANN search. The paper argues explicitly against generative retrieval over semantic IDs.

## Problem
Retrieval and ranking at Pinterest are separate large transformers that encode the same user history, so parameters, training and serving cost are paid twice. Earlier unified models share only parts of the pipeline:
- HSTU needs interleaved item/action tokens for ranking. This doubles the context and breaks input compatibility with retrieval.
- OnePiece was deployed only as a retriever or only as a ranker.
- OneRec, OneRanker and GPR decode semantic IDs end to end and replace the funnel. This loses composability with other candidate sources (keywords, trending) and loses per-stage A/B testing and rollback.

The paper names three obstacles:
- **Usage divergence:** ANN over millions of items versus cross-attention over hundreds.
- **Objective divergence:** sampled softmax versus binary cross-entropy on impressions, where prior unified models needed multi-stage pre-training, fine-tuning and RL.
- **Serving cost:** the unified model must be cheaper than two models while staying modular.

## Method
- **Backbone (PinRec, Sec. 3 background, Eq. 1).**
  - Items are represented by pre-trained OmniSage embeddings plus CLIP-style multimodal embeddings; search queries use pre-trained search embeddings. An MLP projects them to an L2-normalised vector.
  - The model is a causal decoder-only transformer over the history H(u, t_max).
  - Training uses sampled softmax over in-batch and random negatives with a frequency-corrected score s = lambda * q^T i - log Q(i). Q is the item's sampling probability, estimated with a count-min sketch.
  - Retrieval generates query embeddings autoregressively and looks each up in a CPU Faiss index. Recall@10 counts a hit when a target is in the top 10 of *any* generated query embedding.
- **Semantic IDs: none.**
  - There are no RQ-VAE or residual k-means codes, no codebooks, no beam search over codes and no constrained decoding.
  - Sec. 2 cites LIGER (arXiv 2411.18814) and Ding et al. 2026 ("How Well Does Generative Recommendation Generalize?", arXiv 2603.19809). By the paper's account, generative retrieval over semantic IDs "is not a clear win over dense retrieval due to lossy compression, inefficiency of the generation objective, and cold-start failures". Also, with semantic IDs, token relations are memorised in the model's parameters rather than kept in an external index, where embedding proximity is preserved.
  - The design principle is to compose proven parts (ANN, cross-attention ranking, KV cache) rather than decode IDs.
- **Reformulation for ranking (Sec. 3.1).** Three changes:
  - per-candidate action probabilities over a small impression set;
  - binary labels for each action type c in 1..C (click, save, hide, ...) with per-head binary cross-entropy;
  - impression negatives in the data.
- **MAM: input (Sec. 3.1.1).** The action multi-hot a_i goes through a linear layer and is concatenated with the item representation before the input projector.
- **MAM: masking.**
  - History positions are masked independently with probability p_mask.
  - Candidate (future) positions are always masked.
  - A masked position gets a dedicated [MASK] class, an extra one-hot dimension, so "action unknown" is distinct from "no action".
  - Causality guarantee (Eq. 2): z_i depends on Phi_i and on (Phi_t, m_t * a_t) for t < i only.
- **Attention pattern.**
  - The sequence is [history_1..n || candidates_1..k], using the M-FALCON pattern: the history attends causally, and each candidate attends to all of the history and to none of the other candidates.
  - All candidates share position IDs, feedview type and timestamp, for unbiased scoring.
  - Cost falls from O((n+k)^2) to O(n^2 + nk). The block-sparse pattern runs through flex attention.
- **Losses.**
  - L_item: sampled softmax at every unmasked past position (Eq. 3).
  - L_action: a sum over action types of w_c times [mean BCE over masked past positions + mean BCE over the k candidates], with one MLP head h_psi_c per action (Eq. 4).
  - Total: L = L_item + L_action (Eq. 5).
  - The item embedder is trainable and shared, so retrieval and ranking score against identical item vectors.
- **Data (Sec. 3.2).**
  - Each example is a past sequence of (Pin, surface, action, timestamp) tuples plus a future feedview (the full impression slate with labels).
  - Training keeps 10% of non-engaged feedviews. Evaluation uses an unsampled, randomised replay set.
  - User sequences and feedviews live in separate Iceberg tables, hash-bucketed by user ID and joined in memory inside the trainer with Ray. This avoids history fan-out duplication and makes context length and sampling into train-time knobs.
- **Serving (Sec. 3.3, Fig. 3).**
  - The PinRec Triton ensemble does history assembly, then autoregressive retrieval, then a Faiss ANN lookup, followed by a new ranking node.
  - `Faiss.search_and_reconstruct` returns candidate IDs together with their stored embeddings, so ranking needs no embedder pass and no feature fetch.
  - The history KV is written by the retrieval process into a pre-allocated GPU pool of shape [L, S, H, n, D] (layers, slots, KV heads, past length, head dim). The pool is shared through CUDA IPC handles; slots are reused round-robin, and ranking copies a request's KV into its own decode cache.
  - FP8 is used for training and inference (Transformer Engine, E4M3 forward, E5M2 backward) at a cost of -0.5% offline.

## Experiments and results
- **Table 1** (Board More Ideas data; Hit@3 uses the save head; Recall@10 is measured on a different eval set; all rows use the 12-layer MAM backbone):

  | Model | Hit@3 | Recall@10 |
  |---|---|---|
  | TransActV2+DCNv2 (production ranker) | 0.088008 | - |
  | HSTU, interleaved, matched effective length | 0.097326 | 0.76161 |
  | PinRec | - | 0.77486 |
  | PinRec fine-tuned for ranking, embedder frozen to keep the Faiss index valid | 0.095869 | - |
  | UniPinRec without item loss (ranking-only) | 0.097345 | - |
  | **UniPinRec** | **0.10096** | **0.77659** |

  - Joint training beats ranking-only (+3.7% relative), so the retrieval loss helps ranking.
  - Joint training beats pre-train-then-fine-tune with a frozen embedder (+5.3%).
  - Retrieval is not hurt (+0.2%).
- **Table 2** (ranking forward latency on an L40S; B=8, n=992, k=656, L=12):
  - Prefill baseline (bf16 / SDPA / compile): 25.72 ms.
  - Flex attention: 19.70 ms (1.31x).
  - KV-cache decode: 10.42 ms (2.47x).
  - Decode with flex and compile (the setting used online): 8.57 ms (3.00x).
  - FP8 with flex and CUDA graph: 6.56 ms (3.92x).
- **Table 3** (online serving against a bf16 baseline for both stages):
  - bf16 decode with flex and compile: -11.1% end-to-end latency, +63.6% QPS.
  - FP8 decode with flex and graph: +109.1% QPS but +6.7% latency. Transformer Engine needs the leading dimension B*S to be a multiple of 8, which forces batching of the S=1 autoregressive retrieval steps. FP8 is not yet used online.
- **Table 4** (masking ratio p_mask, Hit@3):
  - 0.0: 0.09923
  - 0.1: 0.10078
  - 0.2: 0.10096 (best)
  - 0.3: 0.10075
  - Any masking helps.
- **Figure 4** (scaling, Hit@3; relative FLOPs in brackets):
  - Depth L = 2/4/8/12/24 at S=1024: 0.09584 / 0.09855 / 0.10019 / 0.10096 / 0.10147 (1x to 12x).
  - Sequence length S = 256/512/1K/2K at L=12: 0.09647 / 0.09829 / 0.10096 / 0.10353 (1.3x to 14.2x).
  - Neither axis saturates. Longer history gives the larger gain but costs quadratically.
- **Online A/B, L0+L1 against PinRec retrieval only** (still followed by the production TransActV2 L2 ranker):
  - Board More Ideas (overfetch about 2x, Table 5): surface saves +0.95%, site-wide saves +0.08%. Board was added as a new sequence modality.
  - Notifications (overfetch about 3x, Table 6): push opens +0.91%, push opens for dormant users +1.72%, notification-surface saves +3.84%, email clicks +0.30%, WAU +0.09%.

## Limitations
- No semantic IDs or code-generation baseline is run. The case against semantic-ID generative retrieval is argued from citations, not measured here.
- Single runs only: there are no seeds, variances or confidence intervals. Most Table 4 differences are in the fourth decimal place.
- The two columns of Table 1 use different evaluation sets. Ranking is reported only through the save head's Hit@3. Model width, parameter count and C (the number of actions) are not given.
- The HSTU comparison uses "matching effective sequence length", which disadvantages interleaving by construction.
- Online, only L0+L1 is replaced; the L2 ranker remains. Offline gains are diluted, and the absolute online lifts are about 1%.
- The blended-data details are thin: how the future feedview is chosen, and how the action loss weights w_c were tuned.
- The KV-sharing machinery (CUDA IPC, Triton) is infrastructure-specific.

## Relevance to this workspace
- **Semantic IDs (PLAN row 197).**
  - The paper gives no recipe for building codes: no RQ-VAE, no residual k-means, no codebook sizes, no collision handling, no constrained beam search. Take those from OneRec, TIGER or LIGER instead.
  - What it adds is a caution. Pinterest chose dense ANN over semantic-ID generation, citing lossy compression, a weak generation objective and cold-start failures.
  - For us, a user has a few dozen categories, so there is no corpus-scale retrieval problem that codes would solve. The value of codes is only as a cross-household shared vocabulary: the 64 clusters as level 1, the crowd line "Others filed this payee as", and new users or first-time payees.
  - Row 197 should therefore be judged against the plain dense alternative it competes with: nearest cluster centroid by two-tower embedding, then the user's category in that cluster. If generating the code does not beat that nearest-centroid lookup on blind_v1 and the owner's budget, the code layer adds only lossiness.
- **MAM maps directly onto the history encoders.**
  - Our history rows are (bank string, amount, date -> category). Treat the category as the "action": embed it, concatenate it to the row, and mask it with p≈0.2 using a distinct [MASK] (not "uncategorised"). Train the encoder to predict the masked rows' categories as well as the current one.
  - This gives denser per-position supervision from the same sequences. It also forces reliance on payee and amount content when labels are absent, which is the new-user / first-time-payee case.
  - Cheap test: add a masked-history-label loss to the bge-small InfoNCE training behind kNN/MaxSim (currently 68.4/69.6%).
- **Joint loss over fine-tuning with a frozen embedder.**
  - Table 1's ordering (joint > ranking-only > fine-tune with frozen embedder) argues for training the history encoder with InfoNCE (transaction <-> earlier transaction) plus a per-category classification head jointly, with a trainable shared embedder, rather than freezing an InfoNCE encoder and adding a head later.
- **Frequency-corrected InfoNCE.**
  - The -log Q(i) correction on in-batch negatives (Eq. 1) is a one-line change for our encoders. Popular payees and categories ("Groceries", "Amazon") are over-represented as in-batch negatives. Correct by log batch frequency.
- **Candidates attend to the history but not to each other.**
  - This is our decider's setting: score every user category with one answer slot, given one history prefix. If the slots are currently separate forward passes, or appear in one sequence where they can attend to each other, the M-FALCON mask (shared position IDs, a block-diagonal mask on the candidates) scores all k categories in one pass with order-independent scores, at a cost of O(n^2 + nk).
  - Equivalently, at serving time, cache the KV of the user's prompt prefix (category list plus 24 history rows) and decode only the new transaction and its category slots. Per-user prefix caching is the transferable part of their 2.4-3x speedup.
  - Cross-process CUDA IPC sharing does not transfer: our encoder and decider are different models, so no KV can be shared between them.
- **Retrieval, then ranking, as a gate.**
  - Their L0+L1 overfetch-then-rank is the same shape as "cheap encoder proposes, decider decides". With our tiny candidate set, the analogue is a cheap encoder that auto-files confident cases and passes the shortlist, or the full list, to the 4B model.
  - The paper's gain came from training the two stages jointly. Our fusion gave only +0.3 with settings chosen on synthetic data. Consider training the gate's encoder with the decider's labels or logits as an extra target, rather than fusing two independent models.
- **Scaling.**
  - Sequence length beat depth per FLOP (S 256 -> 2K: +7.3% relative Hit@3). Test a longer history than 24 rows in the decider, and longer histories in the encoders, before adding model size.
- **Does not transfer.**
  - Corpus-scale ANN.
  - The impression-slate negatives: we have no "shown but not chosen" log. The closest analogues are the user's categories that were not chosen, or corrections.
  - FP8 and Triton serving.
  - The engagement metrics.
