# TransAct V2: Lifelong User Action Sequence Modeling on Pinterest Recommendation

- arXiv 2506.02267 (v1, 2 Jun 2025) - https://arxiv.org/abs/2506.02267
- Xue Xia, Saurabh Vishwas Joshi, Kousik Rajesh, Kangnan Li, Yangyi Lu, Nikil Pancha (work done at Pinterest), Dhruvil Deven Badani, Jiajing Xu, Pong Eksombatchai (Pinterest)
- Venue: preprint (acmart sigconf, no conference given; arXiv cs.IR)
- Source: references/papers/2506.02267/paper_flat.tex

## One-paragraph summary
TransAct V2 is the user-sequence module inside Pinterest's Homefeed CTR ranker. Each user has a lifelong action history of about 10^4 actions. For each candidate pin, the model keeps only the history actions nearest to that candidate (by PinSage dot product). It adds the most recent r real-time actions and candidate-nearest actions from the real-time and impression sequences. The resulting sequence of about 192 tokens goes through a tiny causal transformer (2 layers, 1 head, d_model 64, FFN 32), and the candidate's embedding is concatenated onto every token ("early fusion"). Max-pooled outputs feed the ranker's feature-crossing layers. An auxiliary Next Action Loss (sampled softmax: from the causal output at t, predict the positive pin at t+1) uses pins the user saw but did not engage with as negatives. These impression negatives beat in-batch negatives from other users. Offline, the full model adds +13.31% HIT@3/repin and -11.25% HIT@3/hide over a no-sequence baseline, against +7.74%/-6.86% for TransAct V1. Online, against TransAct V1 it gives +6.35% repins, -12.80% hides and +1.41% time spent. Most of the rest of the paper covers serving: nearest-neighbour feature logging, request de-duplication, int8 embeddings and a single fused Triton transformer kernel.

## Problem
CTR rankers in industry use short real-time sequences (about 10^2 actions) because of serving cost, so they lose long-term interests and drift toward echo chambers. The alternatives each have a cost: offline compression of lifelong history (TWIN v2, clustering) is blind to the candidate and loses information, and TWIN needs expensive offline inference and caching. CTR-style sequence encoders also have no next-action objective. The task is to use lifelong (O(10^4)) sequences end to end inside a point-wise ranker, within latency and storage limits.

## Method
- Base ranker: point-wise multi-task wide-and-deep model (Fig. 2). Weighted multi-head binary cross-entropy over action heads, Eq. 1. Final ranking score S = sum_h w_h f(x)_h.
- Sequences:
  - Lifelong S_LL: explicit actions (repins, clicks, hides; no impressions) over years. Maximum length is the 90th percentile of users' 2-year history lengths, weighted by visit frequency.
  - Real-time S_RT and impression S_imp: O(10^2) each.
  - Token features: timestamp, action type (multi-hot if several actions hit the same pin), surface, and a 32-d PinSage embedding.
  - PinSage is stored as int8 by affine quantisation, q = clamp(e/0.65 x 127, -127, 127).
- Candidate-anchored retrieval (Eq. 2-3): S_all = NN(S_LL, c) + S_RT[:r] + NN(S_RT[r:], c) + NN(S_imp, c). NN takes the top K tokens by dot product between the candidate's PinSage embedding e_c and the sequence's PinSage embeddings. The most recent r actions are always kept "regardless of the similarity with the candidate item". |S_all| is "several hundred" at most; 192 was chosen.
- Feature encoding (Eq. 4): F = CONCAT(E_PinSage(S_all), e_c) + E_act + E_surf + E_pos. Positional encoding is learned, and d = d_act = d_surf = d_pos = 2 d_PinSage. Concatenating e_c onto every token is the "early fusion" carried over from TransAct.
- Encoder: 2 layers, 1 head each, d_model 64, FFN 32, causal mask. Output U goes through a linear layer and max pooling into feature crossing (multi-head task), and separately into Next Action Loss.
- Next Action Loss (Eq. 5-7):
  - Loss: sampled softmax L = -log(e^<u(t),p_u(t+1)> / (e^<u(t),p_u(t+1)> + sum_n e^<u(t),n_u>)), summed over users and t, with total L = L_CE + w_NAL L_NAL.
  - Causal mask: prevents leakage.
  - Positives: all tokens of S_RT[:r] with positive engagement.
  - Negatives, two options: (a) in-batch, N pins from another user's sequence in the batch; (b) impression-based, pins from S_imp that this user saw but did not engage with.
- Data and serving pipeline (Sec. 3.4, Fig. 4-5):
  - Logging: NN features are logged instead of full sequences (storage O(L) to O(1)). Training reads the logged NN features; serving fetches the full sequence and runs the NN search on the GPU.
  - Request-level de-duplication: a sparse format with no broadcasting, plus a broadcast-free Triton NN kernel; 8x fewer PCIe bytes.
  - Fused dequantisation, L2 normalisation and NN search: 20% lower latency.
  - Single Kernel Unified Transformer (SKUT): QKV, attention, layernorm and FFN in one Triton kernel held in SRAM, on A10G (weights fit in 6 MB); 6.6x faster forward than PyTorch.
  - Pinned Memory Arena: up to 35% faster inference.
- Data: 6.9B training instances, 182M users, 350M pins, two weeks downsampled; evaluated 7 days after training ends; trained from scratch. Metric: HIT@3 per head within a (user, chunk) ranking request.

## Experiments and results
- Table 2 (offline, HIT@3/repin and HIT@3/hide, lifts over the wide-and-deep model without sequence features):

  | Model | HIT@3/repin | HIT@3/hide |
  |---|---|---|
  | BST (RT) | +6.04% | -0.49% (n.s.) |
  | TransAct (RT) | +7.74% | -6.86% |
  | + NAL in-batch | +8.41% | -8.63% |
  | + NAL imp | +8.92% | -9.08% |
  | TransAct V2 (RT + LL + NAL imp) | +13.31% | -11.25% |

  Read across rows: lifelong history adds about 4.4 points of repin over the best RT-only model; NAL adds 0.7 to 1.2.
- Table 3 (online A/B, 1.5% of traffic per arm, baseline TransAct RT):
  - NAL_imp alone: repins +0.27% (n.s.), hides -6.26%, diversity +0.08% (n.s.), time spent +0.10% (n.s.).
  - TransAct V2: repins +6.35%, hides -12.80%, impression diversity +0.45%, time spent +1.41%. The paper notes that "1% increase in repin volume is considered as a substantial gain".
- Table 4 (NAL negatives): in-batch +0.63% repin / -1.90% hide; impression-based +1.10% / -2.39%.
- Table 6 (NAL weight, impression negatives): 0.0001 +0.58%/-2.09%; 0.001 +0.86%/-2.6%; 0.01 +1.10%/-2.39% (chosen); 0.1 +0.54%/-2.39%. Too large a weight hurts the main task.
- Table 7 (NAL loss type): cross-entropy -1.31% repin / +0.27% hide (n.s.); sampled softmax +0.18% / -1.84%. Cross-entropy as an auxiliary loss hurt.
- Fig. 6: longer |S_all| and a wider FFN raise HIT@3/repin but cost latency. Chosen: length 192, 2 layers, FFN 32.
- Table 5 (SKUT vs PyTorch memory-efficient attention): latency -59.29% at length 64, -85.09% at 192 (the production shape, batch 256), -73.10% at 2048; memory -10% to -20%. Batch sweep (Table 8): -60.36% at batch 16 up to -87.79% at 2048. Against FlashAttention-2 without masks: -66.4% latency, -5.5% memory.
- Serving (Fig. 7-9, from the figure data; the latency figures exist only as pgfplots data inside the tex):
  - Copy time: 85/82/85% lower at batch 128/256/512.
  - Model-run p99 latency: 61/120/215 ms baseline vs 15/29/41 ms with everything on.
  - Heatmap reduction factors (p50/p90/p99): model forward about 2.2x, pin-memory copy 31-46x, queuing delay about 1300x, end-to-end inference 103x/338x/250x.

## Limitations
- Only proprietary Pinterest data; no public benchmark, and the authors say public CTR datasets lack the needed sequences.
- No ablation separating the lifelong sequence from the NN retrieval, varying K or r, or comparing NN retrieval with plain recency over the long history. The +13.31% bundles lifelong history and NAL.
- No comparison with TWIN or TWIN v2 ("expensive offline inference"), HSTU or BERT4Rec.
- NAL ablation tables use a different baseline from Table 2 (the numbers are not on the same scale), and the NAL-only online arm is mostly not significant.
- No absolute metrics or confidence intervals beyond significance stars. The appendix includes a leftover authoring outline.

## Relevance to this workspace
- **What our TransAct-style query (65.0%) dropped.** Our row-195 query is a bi-encoder: 5 recent and 5 nearest earlier filings with ages, compared by cosine with category text. TransAct V2 does two things we did not:
  - The candidate conditions everything. The candidate's embedding is concatenated onto every history token before attention, and the score comes from a small learned network over that candidate-specific sequence, not a dot product.
  - The retrieval is anchored on the candidate.

  In our setting the candidate is a category. The direct port: for each (transaction, candidate category) pair, build a short sequence of
  - the user's filings nearest to the transaction,
  - the r most recent filings,
  - the filings into that category nearest to the transaction.

  Give each token [filing embedding; candidate-category embedding; is-filed-to-candidate flag; cosine to the transaction; age; amount]. Run a 2-layer, 1-head, d=64 causal transformer, max-pool, and output a logit. That is cross-encoder-like scoring at tiny cost, and could close the gap between the 65.0% query and the 68-70% kNN/MaxSim models, which already use candidate-anchored evidence implicitly. This is cheap enough for the 3090.
- **Keep the most recent r regardless of similarity.** This supports the "recent + nearest" design but says to keep both channels separate (S_RT[:r] vs NN(...)), which our row-195 query already does. The hyperparameter that mattered for them was the total length (192), not a 5+5 split. Try 16-64 nearest plus 8-16 recent.
- **Negatives.** Their strongest result transferable to us is Table 4: hard "seen but not chosen" negatives beat in-batch negatives from other users (+1.10 vs +0.63 repin). Our nearest equivalent of an impression is the user's own other categories, which were on screen and not chosen. Our encoders train on in-batch negatives (other households) plus one hard negative. Move to a full softmax over the user's own category list (30-100 options, so no sampling needed), with in-batch negatives as extra. Their Table 7 also argues for softmax over binary cross-entropy in any auxiliary loss.
- **Auxiliary next-filing loss for the planned user tower.** With a causal encoder over filings, predict filing t+1's category from position t using sampled softmax, with weight w about 0.01 relative to the main loss (Table 6 shows a narrow optimum: 0.1 and 0.0001 both lose about half the gain). This is cheap to add to the user-state work (PLAN row 196).
- **Lifelong history is affordable.** Their trick is to run the nearest-neighbour search over the full history at serving time and log only the selected tokens for training. The decider sees only 24 history rows. With 21k transactions on the owner's budget, int8 bge-small vectors are about 8 MB per heavy user, so selection from the whole history is free. This argues for choosing the decider's history rows by similarity over the full history (we partly do this with similar-payee rows), not by recency alone.
- **What does not transfer:** the serving engineering (SKUT, pinned arenas, de-duplication) only matters at about 29k examples per second per host. Repin and hide heads have no analogue for us. Their candidate pool is a shared corpus with fixed PinSage item embeddings, while our candidates are per-user names, so the category embedding must come from the name plus its filings rather than an ID table.

## Key references worth following up
- Xia et al. 2023, TransAct (KDD '23), the V1 design with early fusion and real-time sequences (arXiv 2306.00248, in this folder).
- Pi et al. 2020, SIM (search-based lifelong interest modelling; the source of candidate-anchored retrieval).
- Chang et al. 2023 TWIN; Si et al. 2024 TWIN V2 (compression by clustering).
- Pancha et al. 2022, PinnerFormer (sampled softmax for user representations), arXiv 2205.04507.
