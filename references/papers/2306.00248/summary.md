# TransAct: Transformer-based Realtime User Action Model for Recommendation at Pinterest

- arXiv 2306.00248 (v1, 31 May 2023) - https://arxiv.org/abs/2306.00248
- Xue Xia, Pong Eksombatchai, Nikil Pancha, Dhruvil Deven Badani, Po-Wei Wang, Neng Gu, Saurabh Vishwas Joshi, Nazanin Farahpour, Zhiyuan Zhang, Andrew Zhai (Pinterest)
- Venue: KDD '23 (DOI 10.1145/3580305.3599918); code at github.com/pinterest/transformer_user_action
- Source: references/papers/2306.00248/paper_flat.tex

## One-paragraph summary
TransAct is the realtime sequence module inside Pinnability, Pinterest's Homefeed ranking model (Wide and Deep with DCN v2 crossing and multi-head action prediction). It encodes the user's 100 most recent actions. Each action is a 32-d PinSage embedding of the pin plus a learned action-type embedding. The candidate pin's PinSage embedding is concatenated onto every row (early fusion, "concat"). The result goes through a 2-layer, 1-head transformer encoder with no positional encoding. Output is compressed to the first K=10 output rows plus a max-pool over all rows. Separately, a daily batch user embedding (PinnerFormer) provides long-term interest; the hybrid of the two is best (Table 2). Offline HIT@3/repin is +9.40% and HIT@3/hide is -14.86% against a baseline without sequences (Table 1). Online it gives +11% Homefeed repins and -10% hides (Table 7). A random time-window mask during training restores most of the lost diversity, and the model has to be retrained twice a week.

## Problem
Ranking needs both short-term intent (what the user just did) and long-term interests. End-to-end sequence models capture the first but are expensive with long histories. Batch embeddings are cheap but stale. Earlier sequence models such as BST and DIN treat all actions alike and miss negative actions such as hides. They also use positional encodings, which the authors found unhelpful.

## Method
- Pinnability (Section 3.1, Fig. 2): pointwise multi-task prediction of actions (click, repin, hide, ...). Categorical features use embeddings, numerical features use batch norm, then full-rank DCN v2, then FC layers with one head per action. Loss (Eq. 1) is cross-entropy weighted per head, with w_h = sum_a M[h,a] y_a (Eq. 2; label-weight matrix M, Appendix A, Table 9 example). For instance, a hide raises the weight on the click and repin heads to 100. Each user also gets a weight w_u = w_state * w_location * w_gender.
- Sequence feature (Section 3.2): the most recent |S|=100 pin-level actions, newest first, zero-padded. Per action: timestamp, action type, 32-d PinSage embedding.
- Feature encoding (Section 3.3.1): an action-type embedding table (d_action=32) is concatenated with the PinSage embedding.
- Early fusion (Section 3.3.2): "append" adds the candidate as an extra sequence token with a dummy action type (as in BST). "concat" concatenates the candidate's PinSage embedding to every action row, giving U of shape |S| x (d_action + 2 d_PinSage). Concat was chosen.
- Sequence model (Section 3.3.3): standard transformer encoder, 2 layers, 1 head, FFN d_hidden=32, dropout 0.1. No positional encoding; Appendix B, Table 10 shows learned, sinusoidal and linear-projection PE all make hide worse (+0.78% to +2.29%) with repin about flat.
- Random time window mask (Section 3.3.4): at each training forward pass, sample T uniformly from 0 to 24 hours and mask all actions in (t_request - T, t_request). Not applied at inference. It counters the "rabbit hole" of recommending whatever was just engaged.
- Output compression (Section 3.3.5): take the first K output columns (the most recent actions) and concatenate MAXPOOL(O) to get a vector of size (K+1)d, fed into DCN v2. K=10.
- Hybrid: the TransAct output sits alongside the PinnerFormer batch user embedding and other user features.
- Training: 3 weeks of feed-view logs (2 weeks train, 1 week eval), negatives downsampled to a fixed ratio, 3B instances, 177M users, 720M pins. Adam, 5000-step warmup to LR 0.0048, then cosine; batch 12000.
- Serving (Section 3.4): FLOPs grew 65x (Appendix C, Table 11: 60M to 92M parameters, 1M to 77M FLOPs; CPU latency 22 ms to 712 ms; on GPU 8 ms at cost 1x). This was made possible by fused kernels (cuCollections hash embedding lookup), one combined CPU-to-GPU copy, larger batches and CUDA graphs. Realtime features come from Flink and Kafka into Rockstore.

## Experiments and results
- Evaluation metric: HIT@3 per head, on randomly shuffled sessions so position bias is removed. Results are relative to Pinnability without realtime sequences; all p < 0.05 unless marked.
- Table 1 (repin all / non-core; hide all / non-core):
  - WDL + sequence with average pooling: +0.21% / +0.35%; -1.61% / -1.55%
  - BST with all actions: +4.41% / +5.09%; +2.33% / +3.59% (hide gets worse)
  - BST with positive actions only: +7.34% / +8.16%; hide not significant
  - TransAct: +9.40% / +10.42%; -14.86% / -13.54%
  - Gains are larger for non-core (low-history) users.
- Table 2, hybrid ablation (removing a component): without PinnerFormer, repin -2.46% and hide +3.61%. Without TransAct, repin -8.59% and hide +17.45%. Keeping TransAct and PF but dropping all other user features: only -0.67% and +1.40%.
- Table 3, sequence encoders (repin / hide):
  - average pooling +0.21% / -1.61%
  - CNN +0.08% / -1.29%
  - RNN -1.05% / -2.46%
  - LSTM -0.75% / -2.98%
  - vanilla transformer on PinSage only +1.56% / -8.45%
  - Most of TransAct's +9.40% therefore comes from action types, early fusion and output compression, not the transformer itself.
- Fig. 4: performance grows sub-linearly with sequence length, and concat beats append at every length.
- Fig. 5: 4 layers with FFN 384 is best but adds 30% latency; 2 layers with FFN 32 was kept.
- Table 4, output compression (repin / hide):
  - random column +6.80% / -10.96%
  - first column +7.82% / -11.28%
  - random K columns +7.42% / -12.12%
  - first K columns +9.38% / -14.33%
  - all columns +8.86% / -15.70%
  - max pool +6.38% / -14.15%
  - first K + max pool +9.41% / -14.86% (chosen)
  - all columns + max pool +8.67% / -12.64%
- Online (Table 7, 1.5% traffic per arm): repins +11.0% (non-core +17.0%), hides -10.0% (non-core -10.5%), time spent +2.0% (non-core +1.5%).
- Fig. 6: without retraining, the gain decays over 2 weeks. Production retrains twice a week.
- Impression diversity (unique top-level interests) fell 2-3% with TransAct. The random time-window mask brings it back to -1% with repins unchanged. Higher dropout and random action masking raised diversity but cost engagement.
- Other surfaces (Table 8): Related Pins repins +2.8%, Search repins +2.3%, notification email CTR +1.4%, push open +1.9%.
- Section 6.1: full-traffic gains exceeded the A/B numbers, which the authors attribute to a positive feedback loop.

## Limitations
- Private data only; no public benchmark comparisons. All numbers are relative lifts with no absolute HIT@3.
- Rows are a 32-d precomputed content embedding (PinSage) that is not fine-tuned, so the transformer works on frozen item vectors.
- The "no positional encoding" finding rests on a newest-first sequence feeding into a "first K" output read-out, which already supplies recency implicitly. Timestamps are used only for the mask, not as features.
- Gains depend on retraining frequency and on feedback loops that offline A/B tests underestimate.
- It is a ranking module crossed with hundreds of other features, not a standalone retriever. Its value alone is not measured.

## Relevance to this workspace
- Our "TransAct-style" query scored 65.0%, the lowest of the history-aware encoders. That query put the 5 most recent and 5 nearest earlier filings, with ages, into one text against a category text. It lacks the parts that carry TransAct's gain. Table 3 shows the bare transformer adds only +1.56% repin; the rest comes from:
  1. Action-type embeddings. Our analogue is the filed category on each history row. Embed it as a learned or text vector per row, not as words in a long string.
  2. Concat early fusion. Concatenate the candidate transaction's embedding, and ideally the cosine and amount ratio, onto every history row so attention is target-aware (DIN-like).
  3. First-K plus max-pool read-out into a crossing layer, not a single pooled bi-encoder vector.

  A faithful test is a small 2-layer, 1-head transformer over frozen bge-small vectors. Rows would be [hist_txn_emb, hist_category_emb, cand_txn_emb, cos, log-age, amount features] for 20-50 earlier filings. Its output plus a candidate-category embedding goes to a per-category score with softmax over the user's categories. It is cheap enough for a 3090.
- The hybrid finding (Table 2) matches our component plan. TransAct is the realtime part; the per-user state vector or user tower (row 196) is the "batch PinnerFormer" part. Their result is that the sequence matters about 3.5x more than the batch embedding (8.59 vs 2.46), but both contribute. Expect the state vector to add a little on top of history encoders, not to replace them.
- Random time-window mask: our analogue is randomly hiding the most recent or same-payee filings during training. That keeps the model from simply copying the last label for a payee and teaches the non-exact-match cases that matter for first-time payees. It is a training-only augmentation with zero inference cost, and worth one arm.
- Positional encoding gave nothing for them, which supports feeding ages as numeric features (as our query did) rather than as positions. Note that their "first K = most recent" read-out is itself a recency prior.
- Gains were largest for non-core (thin-history) users. On the owner's budget, report the effect for categories with few filings separately.
- Does not transfer: GPU-serving engineering, the label-weight matrix for multiple actions (we have a single label), feedback-loop effects, and retraining cadence beyond "users' categories drift; re-index history".
