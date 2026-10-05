# PinFM: Foundation Model for User Activity Sequences at a Billion-scale Visual Discovery Platform

- arXiv 2507.12704 (v3, 20 Aug 2025; v1 17 Jul 2025) - https://arxiv.org/abs/2507.12704
- Xiangyi Chen, Kousik Rajesh, Matthew Lawhon, Zelun Wang, Hanyu Li, Haomiao Li, Saurabh Vishwas Joshi, Pong Eksombatchai, Jaewon Yang, Yi-Ping Hsu, Jiajing Xu, Charles Rosenberg (Pinterest)
- Venue: preprint (acmart nonacm template with placeholder conference fields; no venue stated)
- Source: references/papers/2507.12704/paper_flat.tex

## One-paragraph summary
PinFM pretrains a GPT-2 style (Pre-LN) decoder-only transformer on two years of Pinterest user activity sequences. Each event is (timestamp, action type, surface, item ID). Items are represented by large hashed ID-embedding tables of about 20B parameters, not by content embeddings. The objectives are InfoNCE-based:
- next positive item;
- every positive item in a future window (multi-token);
- a future window from the position equal to the downstream sequence length.

The model is then fine-tuned inside existing ranking models (Home Feed, Related Items) at about 1/10 of the ranker's learning rate. The candidate item is appended to the user sequence ("early fusion") so the transformer can cross-attend between history and candidate. Early fusion gave +2.9 to +3.8% Save HIT@3 on Home Feed against +1.9% for late fusion with a pooled user vector (Table 1). Without fine-tuning the gain is nearly zero (+0.10%, Table 6).

Cold-start items need extra handling, or the ID-heavy module hurts them (-24% on items younger than 7 days):
- randomise the candidate ID 10% of the time;
- apply item-age-dependent dropout on the module outputs;
- add the GraphSAGE embedding and a learnable token.

With all of these, HF 28d rises from -4.4% to +17.7% (Table 2). DCAT, a once-per-user KV cache with cross-attention for candidates, raises serving throughput 600% and training throughput 200%; int4 embedding quantisation is neutral online. Online: Sitewide Saves +1.20% (HF) and +0.72% (I2I).

## Problem
Large sequence models in recommendation (HSTU, TIGER, TWIN-V2) are trained standalone per application, which is too costly across many applications, and distillation from a big teacher is slow to iterate. A shared, pretrained user-sequence model has to:
- plug into existing rankers that carry many other features;
- score millions of items per second under latency and cost budgets;
- capture user-candidate interactions;
- handle items unseen at pretraining time.

## Method
- Pretraining data (Section 3.1): per-user chronological events S_i = [t_i, a_i (action), v_i (surface), id_i], capped at 16,000 events and cut into non-overlapping training segments of length L ("a few hundred").
- Forward pass (Eq. 1): H = phi_out(M(phi_in(E + V + A))).
  - E, V, A are the item-ID, surface and action embeddings, summed.
  - phi_in and phi_out are pointwise MLPs with L2 normalisation.
  - M is a GPT-2 Pre-LN decoder. HSTU was tried and gave similar results.
  - Timestamps are listed as input data, but how they enter Eq. 1 is not specified.
- Loss (Eq. 2): InfoNCE l(H_i, z) with sim = inner product, a learnable temperature tau (small initial value), and K in-batch negatives that exclude items the same user engaged with positively. The target is z = psi(emb(id)), with psi an MLP plus L2 normalisation.
- Objectives:
  - L_ntl: next-token loss, applied only where the next action is in the positive set A_pos.
  - L_mtl: multi-token loss; H_i predicts every positive item in the window (i, i+L'], subsampled for cost.
  - L_ftl: future-token loss; only H_{L_d} predicts the positives in (L_d, L_d + L']. L_d is the downstream real-time sequence length, smaller than L. The authors compare this to instruction tuning: the model is trained hardest at the length it will be used.
- Fine-tuning (Section 3.2):
  - The pretrained transformer and embedding tables become the user-sequence module of a DLRM/DCN-style multi-task pointwise ranker.
  - Late fusion: sequence to user vector, crossed later. It can be cached per request but has no candidate context.
  - Early fusion (chosen): append the candidate to the sequence. Outputs are the candidate-position embedding (a user x candidate cross) and the pretrained candidate ID embedding.
  - Extra candidate embeddings (content, GraphSAGE) can optionally be projected and summed into the candidate input token, with an alignment loss.
  - Optional L_ntl / L_mtl are kept during fine-tuning.
  - The learning rate of the pretrained module is about 1/10 of the ranker's.
  - The downstream ranking losses are also applied directly to the module's outputs, plus an MSE loss aligning the module's predictions with the final ranker's.
- Cold start:
  - Candidate item randomisation (CIR): replace the candidate ID with a random one 10% of the time.
  - Item-age-dependent dropout (IDD): p = 0.7 on module outputs when the item is under 7 days old, p = 0.5 between 7 and 28 days.
  - GSLT: add the GraphSAGE embedding to the candidate token, plus a learnable token placed before the candidate.
- Design notes:
  - Causal attention is kept in fine-tuning; bidirectional costs -1.4%, attributed to distribution shift from pretraining.
  - ID embeddings were chosen over pretrained content embeddings for storage and serving volume (hundreds of times smaller) and because pretrained ID embeddings transfer (Hsu et al. 2024).
- Efficiency (Section 4):
  - DCAT: unique user sequences are about 1:1000 of candidates in serving and 1:10 in training.
    - The context transformer runs once per deduplicated user (B_u about B/16) and stores per-layer K and V.
    - Each candidate token cross-attends to Psi^{-1}(K_u) || K_c (Eqs. 3-4), with Triton kernels.
    - Throughput: +600% serving and +200% training against FlashAttention self-attention.
    - A fixed length of 256 with KV rotation instead of concatenation, and skipping the last layer's context self-attention at serving, add +25%.
  - Embedding tables: 8 hashed sub-tables of 80M rows x 32 dims, concatenated to 256-d fp16 per ID, about 20B parameters, sharded with TorchRec for training and served from a CPU host.
    - Post-training min-max quantisation with FBGEMM: int4 shrinks each vector from 512 to 160 bits (31.25%) with 7.8% relative L2 error; int8 gives 0.45%.
    - Offline Save -0.06% with int4; neutral in a 2-week A/B test; API latency -7%.

## Experiments and results
- Setup: pretraining on 2 years of activity; fine-tuning on 3 weeks of Home Feed (HF) or Related Items (I2I) ranking data. Metric: relative lift in HIT@3 (actions among the top 3 recommended), Save up is good, Hide down is good.
- Input sequence variants (Table 1), Save HIT@3 for HF / I2I:
  - base (early fusion): +2.91 / +1.76%.
  - plus GraphSAGE: +3.08 / +1.92%.
  - plus GraphSAGE and learnable token: +3.76 / +1.92%.
  - lite-mean (late fusion, mean pool): +1.87 / +1.53%.
  - lite-last (last token): +1.93 / +1.49%.
- Cold start on HF Save (Table 2), overall / items under 28d / under 7d:
  - none: +3.36 / -4.40 / -23.98%.
  - CIR: +3.43 / +1.25 / -4.38%.
  - CIR + IDD: +3.49 / +10.71 / +8.16%.
  - CIR + IDD + GSLT: +3.76 / +17.72 / +12.01%.
- Losses (Table 3), against L_ntl in both pretraining and fine-tuning, Save / Hide:
  - pretrain ntl+mtl: +0.42 / -1.27%.
  - pretrain ntl+mtl+ftl: +0.95 / +2.43% (Hide worse).
  - same pretraining with no fine-tune sequence loss: +0.41 / +1.06%.
  - same pretraining, fine-tune with ntl+mtl: +1.01 / +1.03%. mtl is not used by default in fine-tuning because of memory and time.
- Positive-action definition (Table 4), against Save only:
  - Save + Download: +0.21 / -1.63%.
  - Save + Clickthrough: +0.02 / -2.67%.
  - All - Hide: -0.2 / -1.76%.
  - All - Hide - Clickthrough: +0.19 / -4.1%.
  - The paper's conclusion: "selecting positive actions optimally is a non-trivial problem".
- Pretraining length (Figure 3): 0 to 640k iterations. Save and Hide improve roughly monotonically. They chose about 3 epochs and saw no one-epoch overfitting.
- Fine-tuning matters (Table 6), against no PinFM:
  - frozen PinFM: Save +0.10%, Hide +2.56% (worse).
  - fine-tuned: Save +3.76%, Hide -2.77%.
- Vocabulary size (Table 7), Save / Hide against 20M rows: 40M +0.81 / -0.34%; 80M +0.91 / +0.31%; 160M +1.98 / +0.92%.
- Online A/B (Table 8, all significant at 95%), HF / I2I:
  - Sitewide Saves +1.20 / +0.72%.
  - Surface Saves +2.60 / +2.09%.
  - Fresh Saves +5.70 / -0.82%. I2I shipped without the cold-start fixes. HF without them showed about -5% Fresh Saves, and about +10% with them.
  - The baseline is already a TransAct transformer, with under 0.2% of PinFM's parameters. Feed diversity increased.

## Limitations
- Results are relative lifts on internal data only, with no public benchmark, no absolute values and no variance across seeds.
- The design depends on ID embeddings, so it is weak on new items by construction. Much of the paper is about repairing that (Table 2).
- Gains from pretraining alone are nearly nil without fine-tuning (Table 6).
- Some ablations trade one metric against another (Hide worsens with L_ftl and with some action sets), with no principled rule.
- How timestamps enter the model, the length L, the window L', the weights between losses, and the non-embedding transformer size are not reported.
- There is no comparison with training the same architecture from scratch on the downstream data alone at equal compute, beyond the "0 iterations" point in Figure 3.

## Relevance to this workspace
How it works, mapped to us:
- Pretrain a causal transformer whose tokens are a user's events, with InfoNCE against the embedding of the next positive item.
- Add the multi-token window loss (predict all positives in the next L' events) and the future-token loss at exactly the downstream history length.
- Then insert the pretrained model into the task model:
  - append the candidate as the last token (early fusion);
  - fine-tune at 1/10 of the learning rate;
  - keep the next-token loss as an auxiliary;
  - apply the task loss directly to the module output.

For us the event becomes a filed transaction: payee embedding + amount features + date features + category, summed after projection. The candidate becomes the new transaction appended at the end, with its category to be predicted.

Does a small version make sense? Yes as a contained experiment, with these changes:
1. Use content, not IDs, for payees. PinFM chose IDs for storage and serving reasons that do not apply to us, and paid for it on cold start (Table 2: -24% on 7-day items before the fixes). Our hard case is first-time and unseen payees. Represent each token with frozen bge-small (or our InfoNCE-tuned encoder) payee text, plus a learned embedding for the 64 cross-household clusters where known.
2. Categories are per-household, so there is no shared vocabulary to predict over.
   - Score the next filing's category as InfoNCE over that household's own categories: each category's vector = its name embedding plus the mean of its earlier filings (the MaxSim/prototype view), against the transformer output at the candidate position.
   - Add an auxiliary InfoNCE over the 64 shared clusters, which do generalise across households.
   - Pretraining over many synthetic households then teaches generic routing:
     - recurring amounts and dates;
     - "same payee, same category unless amount differs";
     - trips as bursts of filings to a temporary category;
     - person-named categories tied to transfers.
3. Use L_ftl with L_d equal to our inference history (24 rows, matching decider's prompt, or 5 + 5 as in the TransAct-style query). The paper finds training at the downstream length adds +0.5% over ntl+mtl.
4. Early fusion. Our TransAct-style query (65.0%) is a late-fusion, pooled bi-encoder: history and category text meet only at a dot product. PinFM's largest single gap is early against late fusion (+3.76% against +1.9%). The early-fusion version is a small transformer reading [history tokens..., candidate token] and scoring categories from the candidate position. Expect it to beat the 65.0 TransAct arm. Whether it beats kNN or MaxSim (68 to 70) is open.
5. Fine-tuning is what makes it pay (Table 6: frozen +0.10%). Plan pretraining followed by supervised fine-tuning on the categorisation loss. Do not pretrain and use frozen features in the fusion with decider; our fusion evidence (tuned on synthetic data gives +0.3) already shows that frozen side signals transfer poorly.
6. Scale: a 4-6 layer, d = 256-512 decoder over sequences of a few hundred filings, trained on tens of thousands of synthetic households, fits the 3090 in hours. The parameter count is in the transformer, not in 20B of ID tables. Run on Modal only if sweeping.

Risks specific to us:
- Pretraining on synthetic households learns the generator's regularities. The paper's gains come from 2 years of real behaviour, and our record says synthetic-tuned settings transfer weakly to the owner's budget. Judge on blind_v1/v2 and the owner's budget, and compare against an identical model trained from scratch on the fine-tune data (the "0 iterations" point in Figure 3).
- We have one real household (about 21k transactions, private, never in training), so real pretraining is impossible for now. With more than 1M real users this design becomes much more attractive.
- PinFM's gains are 1-4% relative on top of a strong ranker. Expect the sequence module to add a point or two to decider-4B at best. Its value is as a cheap per-user state (the planned "per-user state vector", row 196: the hidden state at the last filing) and as a non-LLM fallback.

Smaller, reusable details:
- In-batch negatives exclude the same user's other positives. For us: exclude same-household filings of the same category from the negatives.
- The choice of positive actions matters (Table 4). The analogue is which filings count as targets: skip transfers, splits and auto-imported rule filings.
- Cold-start tricks for new categories:
  - randomly mask the candidate payee's identity or embedding 10% of the time;
  - apply dropout on history-derived features for payees seen fewer than n times, so the model learns to fall back on content and crowd signals.
- DCAT's idea (encode the history once, then cross-attend each candidate) is what decider's single answer slot over all categories already does. Reuse the history KV cache across a household's pending transactions.

Does not transfer:
- 20B-parameter ID tables, TorchRec sharding, int4 FBGEMM quantisation, CPU embedding hosts;
- the action and surface vocabulary;
- HIT@3 feed metrics.
