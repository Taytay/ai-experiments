# PinnerFormer: Sequence Modeling for User Representation at Pinterest

- arXiv 2205.04507 (v1, 9 May 2022) - https://arxiv.org/abs/2205.04507
- Nikil Pancha, Andrew Zhai, Charles Rosenberg (Pinterest); Jure Leskovec (Stanford University)
- Venue: KDD '22 (the acmart header gives KDD '22; the arXiv comment says "submitted to KDD '22"; TransAct V2 cites it as KDD '22, pp. 3702-3712)
- Source: references/papers/2205.04507/paper_flat.tex

## One-paragraph summary
PinnerFormer is a single 256-d user embedding computed once a day in batch. A causal PreNorm transformer reads the user's last M=256 pin engagements, each a PinSage embedding plus action type, surface, duration and Fourier time features. The key change is the "dense all action" objective. Instead of predicting the next action, the model predicts, from randomly chosen sequence positions, a random positive engagement within the next K=28 days. A daily-computed embedding then stays useful for two weeks: the drop from realtime to once-only inference is 8.3% vs 13.9% for the SASRec objective (Table 2). It is trained with sampled softmax, a learned temperature, logQ sample-probability correction, and mixed negatives (in-batch plus random). Recall@10 against 1M random pins is 0.229, vs 0.046 for the 20-cluster PinnerSage oracle and 0.198 for a SASRec-objective model. In A/B tests in Homefeed ranking: repins +7.5%, close-ups +6%, time spent +1%, DAU +0.4%. As a new Ads feature: CTR +7% to +10%.

## Problem
Sequential user models are usually realtime: either expensive stateless recomputation after every action, or stateful streaming with fragile hidden state. Pinterest also has tens of ranking models and wants one shared user feature. PinnerSage's 20+ 256-d embeddings per user are too large to put in ranking training data. The goal is one embedding, computed offline once a day, that predicts long-term (14-day) positive engagement.

## Method
- Inputs: the user's last M actions (positive engagements on pins over the past year: saves, clicks, reactions, comments). Each token has:
  - a PinSage 256-d embedding;
  - learnable embeddings for action type and surface (out-of-vocabulary tokens dropped);
  - log(duration);
  - time features.
- Time encoding (Appendix A.1): absolute timestamp, time since the latest action, and gap to the previous action. Each is encoded Time2vec-style with fixed periods and learned phases, cos(2 pi t / p_i + phi), sin(...), plus log(t):
  - absolute time: periods 0.25h, 0.5h, 0.75h, 1h, 2h, 4h, 8h, 16h, 1d, 7d, 28d, 365d;
  - relative times: 32 periods evenly spaced on a log scale from 1 s to 4 weeks.
- Architecture (Eq. 3, Appendix A.2):
  - Linear projection to hidden size H, plus a fully learnable positional encoding.
  - L PreNorm blocks of MHSA with a causal mask and a 2-layer FFN (4H); 8 heads.
  - Output: LayerNorm, MLP (H to 4H to D) with GELU, then L2 normalisation, at every position. e_1 (the most recent) is the user embedding.
  - Pin tower: an MLP on PinSage, L2-normalised.
  - Both towers are learned jointly; users and pins are compared by cosine.
  - Final: M=256, D=256.
- Loss (Eq. 1-2): sampled softmax with learned temperature tau >= 0.01. LogQ correction subtracts log Q_i(v), the probability that pin v is in the batch given user i, estimated with a count-min sketch. Each user on a GPU gets equal weight. (Eq. 2 in the source has a typo: s(u_i, u_i) for s(u_i, p_i).)
- Negatives: in-batch (all positives in the batch, masking the user's own positives), random (uniform from the Homefeed corpus), or mixed (one pool). Negatives are gathered across GPUs: in-batch capped at 5000, random fixed at 8192. There are no explicit negative engagements (no hides).
- Objectives (Fig. 2):
  - Next action: predict A_{T+1} from e_1.
  - SASRec: predict the next positive at every position.
  - All action: e_1 predicts up to 32 sampled positives in the next K days.
  - Dense all action (chosen): random positions s_i, each e_{s_i} predicts one random positive from its own next K days, with a causal mask. Without subsampling this would be 128 x 256 x 32 = 1,048,576 pairs per GPU.
- Positives: Homefeed only (repin, close-up over 10 s, click-through over 10 s), one multi-task embedding with no per-task heads or weights.
- Dataset: one row per user timeline. Sub-sequences and labels are sampled on the fly, so these can be tuned without regenerating data: maximum length, fraction of sequences sampled, maximum sequences per user, and maximum labels per sequence.
- Serving: daily incremental inference for users active in the past day, merged with yesterday's embeddings and pushed to a KV store. Pin embeddings are recomputed daily into an HNSW index.
- Evaluation: Recall@10 of all positives in (t, t+14d] from the embedding at t, against 1M random pins, on users disjoint from training. Diversity metrics: Interest Entropy@50 over about 350 topics, and P90 Coverage@10.

## Experiments and results
- Table 1:
  - PinnerSage oracle, 5 clusters: R@10 0.026, entropy 1.69, coverage 0.130.
  - PinnerSage oracle, 20 clusters: 0.046, 2.10, 0.133.
  - PinnerFormer: 0.229, 1.97, 0.042.
  - PinnerSage keeps the edge on diversity and coverage.
- Table 2 (R@10, SASRec vs PinnerFormer):
  - Once: 0.198 vs 0.229.
  - Daily: 0.216 vs 0.243.
  - Realtime: 0.251 vs 0.264.
  - Going from realtime to batch costs -13.9% for SASRec and -8.3% for dense all action. PinnerFormer beats the next-action model even in realtime.
- Table 3 (objectives, R@10 / coverage):
  - Next Action 0.186 / 0.050.
  - SASRec (softmax) 0.198 / 0.048.
  - All Action 28d 0.224 / 0.028.
  - Dense All Action 14d 0.223 / 0.043.
  - Dense All Action 28d 0.229 / 0.042.
  - A 28-day training window beats 14 days even on the 14-day evaluation (more labels per sequence). Summing objectives did not beat any single one.
- Table 4 (negatives x SPC, R@10 / coverage):

  | Negatives | No SPC | With SPC |
  |---|---|---|
  | random | 0.136 / 0.002 | 0.139 / 0.001 |
  | in-batch | 0.071 / 0.163 | 0.167 / 0.119 |
  | mixed | 0.138 / 0.083 | 0.229 / 0.042 |

  Random negatives alone collapse: about 1000 pins make up 90% of retrievals for 100k users. LogQ does nothing for uniform negatives but more than doubles in-batch recall.
- Table 5 (multi-task): single-task models win their own action type (closeup 0.27, click 0.49, repin 0.17). The multi-task model is second on each (0.23, 0.28, 0.13) and best overall (0.23 vs 0.17, 0.12, 0.13).
- Table 6 (leave one feature out, R@10): PinSage 0.142 (coverage 0.0005), timestamp 0.210, surface 0.224, action type 0.226, duration 0.226, positional encoding 0.228, none 0.229.
- Fig. 4 (sequence length): roughly constant gains per doubling up to about 32, then diminishing. 256 was chosen because 512 needs 16 GPUs and 1024 needs 32.
- Fig. 5 (embedding size): diminishing past 128; small dimensions memorise popularity.
- Table 7 (capacity, R@10): 2x256 0.2189 up to 6x768 0.2293. Bigger is better but by small margins. Head count had no effect.
- Table 8 (SASRec variants): binary cross-entropy 0.138; sampled softmax 0.181; sampled softmax with equal e_1 weight 0.198.
- Sparsifying histories by removing weak engagement gave "no significantly positive results".
- Online:
  - Homefeed, replacing the PinnerSage aggregate (Table 9): time spent +1%, DAU +0.4%, WAU +0.12%, repins +7.5%, click-throughs +1%, close-ups +6%; no decay over several months.
  - Ads, added as a feature (Table 10): CTR +7.1% / +7.3% / +10.0% and gCTR +6.9% / +5.2% / +10.1% (Related Pins / Search / Homefeed).

## Limitations
- Only internal data; the only baselines are SASRec and PinnerSage. Evaluation is retrieval against 1M random pins, which is easier than ranking against impressions.
- A gap to realtime remains (0.243 daily vs 0.264 realtime).
- One embedding loses per-user diversity relative to multi-embedding PinnerSage (entropy 1.97 vs 2.10).
- Only positives are modelled; hides and negative feedback are unused.
- Sequences longer than 256 were not explored because of resources; the comparison would also change the negative pool.

## Relevance to this workspace
- **Dense future-window objective for the planned user state vector (PLAN row 196).** This paper is the reference design. Train a causal encoder over a user's filings. From random positions, predict a random category filed in the next K days (not just the next filing), with sampled softmax and causal masking. Their result that a 28-day window beats 14 days even when evaluating at 14 days suggests a wide window, about one month for a budget (monthly bills, a trip in progress). This matters if user state is computed nightly in batch: the objective is what made the stale embedding hold up (-8.3% vs -13.9%).
- **LogQ correction is the cheapest fix to try on our encoders.** Our in-batch negatives come from other households, so common category names ("Groceries", "Gas") appear as negatives far more often than rare personal ones. Table 4 shows uncorrected in-batch negatives at 0.071 vs 0.167 corrected and 0.229 mixed with correction. Estimate each category text's in-batch frequency (a count is fine at our scale) and subtract log Q from its logit. Also add random negatives: in our case, category names sampled from the cross-household vocabulary. Random-only negatives collapse, so do not use them alone.
- **Time and amount features.** They encode time with fixed-period Fourier features (12 absolute periods including 1d, 7d, 28d, 365d) and log-spaced relative periods from 1 s to 4 weeks, plus log values. Timestamp was the second most important feature (0.229 to 0.210 when removed). Recurring transactions in budgets are periodic (weekly, biweekly pay, monthly bills, annual renewals), so use the same scheme for date and days-since-last-filing-to-this-category in the user tower. "Amounts as numbers" (row 200) could likewise use log|amount| plus sign plus a few log-spaced Fourier periods, rather than amount text.
- **Single embedding vs set.** A single learned sequence embedding beat the PinnerSage cluster oracle 5x on recall, but lost on diversity. Our MaxSim and kNN encoders are the set-based approach. The planned user tower would be the single-vector approach, and the paper suggests it can add to them, not replace them.
- **Smaller points:**
  - Multi-task with no heads (Table 5): a single objective over all positives is the robust choice.
  - Capacity matters little (Table 7): the user tower can be small enough for the 3090.
  - Diminishing returns past about 32 actions (Fig. 4): a user tower over the last 32-64 filings is probably enough.
  - Their evaluation on users disjoint from training matches our rule of holding out users.
- **What does not transfer:** their item side is a fixed shared corpus (PinSage plus an MLP). Our "items" are each user's own category names, so the pin tower becomes a category tower over the name text plus the category's filing history. Their objective has no "choose among the user's own K options" structure. The final scorer for us should still be a softmax restricted to that user's categories.

## Key references worth following up
- Kang and McAuley 2018, SASRec (dense next-item objective).
- Yi et al. 2019, sampling-bias-corrected neural retrieval (logQ); Yang et al. 2020, mixed negative sampling.
- Kazemi et al. 2019, Time2vec.
- Pal et al. 2020, PinnerSage, arXiv 2007.03634.
