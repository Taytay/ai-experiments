# Mass-Editing Memory in a Transformer (MEMIT)

- arXiv 2210.07229v2 (1 Aug 2023) - https://arxiv.org/abs/2210.07229
- Kevin Meng, Arnab Sen Sharma, Alex Andonian, Yonatan Belinkov, David Bau (MIT CSAIL, Northeastern, Technion)
- Venue: ICLR 2023
- Source: docs/papers/2210.07229/paper.txt

## One-paragraph summary
MEMIT extends ROME from single-fact rank-one edits to batch insertion of thousands of (subject, relation, object) memories by directly solving a least-squares update to the MLP down-projection weights of a range of mid-layer MLPs identified by causal tracing. On GPT-J (6B) and GPT-NeoX (20B) it inserts 10,000 facts while keeping paraphrase generalization near 90%, specificity within about 10 points of the unedited model, and generation fluency essentially unchanged, whereas ROME collapses beyond roughly 32 edits, MEND beyond 6, and fine-tuning with weight decay destroys fluency (Table 2, Figure 5).

## Problem
Prior editors (constrained FT, hypernetworks KE/MEND, SERAC, ROME) handle at most tens of edits; SERAC reached 75. Practical use needs hundreds to thousands of simultaneous updates with generalization (rephrasings still recall the fact), specificity (unrelated subjects unchanged) and fluency (no degenerate repetition) (Section 3).

## Method
1. Causal tracing on GPT-J locates a range R of critical MLP layers at the last subject token (R = {3..8} for GPT-J, {6..10} for NeoX; Figure 3, Appendix A).
2. Each MLP down-projection W_out is modelled as a linear associative memory W K = M. Inserting u new key-value pairs while preserving old ones gives the closed form Delta = R K1^T (C0 + K1 K1^T)^-1 (Eqn 14), where C0 = lambda E[k k^T] is an uncentered covariance estimated from 100,000 Wikitext samples (lambda = 15,000 for GPT-J; Appendix B.4).
3. For each edit, a target hidden vector z_i at the last subject token of the top layer L is found by 20-25 gradient steps maximizing log P(o_i | x_j + p(s_i, r_i)) over random prefixes x_j (Eqn 16). Keys are computed from the prefix plus the subject only (Eqn 19), i.e. the key does not depend on the relation.
4. The residual z_i minus h_i^L is spread evenly over the layers in R, re-collecting activations after each layer update (Algorithm 1, Figure 4).
Edit requests must not conflict on the same (s, r) (Eqn 5).

## Experiments and results
- zsRE, 10,000 edits on GPT-J (Table 1): MEMIT efficacy 96.7, paraphrase 89.7, specificity 26.6 (unedited 27.0); FT-W 69.6 / 64.8 / 24.1; MEND 19.4 / 18.6 / 22.4; ROME 21.0 / 19.6 / 0.9.
- CounterFact scaling to 10,000 counterfactual edits (Figure 5, Table 2; the extracted table is garbled, assignments follow the text): MEMIT on GPT-J score 85.8, ES 98.9, PS 88.6, NS 73.7 (unedited 83.5), fluency GE 619.9 (unedited 622.4), reference score 40.1. FT-W attains ES 99.4 but GE 293.9, i.e. "complete generation failure". ROME degrades from n = 32, MEND from n = 6. GPT-NeoX-20B: ES 97.2, PS 82.2, NS 70.8.
- Relation categories (Figure 6): some relations (athlete plays sport P641, owned by P127) are hard for every editor; MEMIT still leads.
- Mixing two relations up to 700 edits gives performance close to the average of the separate relations, so diversity of the batch neither helps nor hurts (Figure 7, Appendix D).
- Ablations (Appendix F): more critical layers improve all metrics; late-layer or attention edits are much worse; specificity and fluency rise monotonically with lambda while efficacy falls, optimum near lambda ~ 10^4 (Figure 13).
- Demonstrations (Appendix E): November 2022 election results inserted with 100% ES and 94% PS; 289 (star, constellation) tuples raise GPT-J from 53% to 86%.
- Runtime, 10k edits on GPT-J (Appendix B): MEND 98 s, FT 0.48 hr, MEMIT 7.44 hr (6.54 hr serial z optimisation plus 0.90 hr matrix update), ROME 12.29 hr. GPT-J fits one 48 GB A6000; NeoX needs two.

## Limitations
Only directional (s, r, o) associations; the paper states explicitly that the reverse association ("The CEO of Apple is Tim Cook") must be inserted separately, and spatial, temporal, procedural and symmetric knowledge are out of scope (Section 6). Specificity still drops about 10 points at 10k edits. Evaluated only on GPT-J and GPT-NeoX; evaluation metrics are probability comparisons against the true object, not generation. Because keys depend only on the subject (Eqn 19), several relations on the same subject share one key; the paper never tests that regime.

## Relevance to this workspace
Informs BASE-2, EVAL-5, TRAIN-4, REAL-3, EVAL-1, LIT-1.
- New subjects: keys and target vectors are computed from actual activations, so no prior representation is needed; zsRE, the election demo and the star-constellation demo insert facts the model largely lacked. But every subject tested is a real-world name; fictional morphological names ("Javish") are untested.
- Scale: 10k edits on 6B with about 10 points of specificity loss; 680 relations is well inside the range. Our 136 species x 5 attributes is exactly the untested shared-key regime: with keys from subject only, five separate edits per species would be averaged by Eqn 14. BASE-2 must either optimise one z per species jointly over its five prompts (a change to EasyEdit's compute_z) or treat each species as one "profile" edit.
- Metric mapping: efficacy ~ L1 recall in trained format; paraphrase ~ bare-format L1 (the 20.6 vs 100 gap in EVAL-1 is exactly the efficacy/paraphrase gap MEMIT's random-prefix objective targets); neighborhood specificity ~ other species' attributes; generation entropy has no analogue in our ladder (EVAL-3 would supply it). Nothing in MEMIT maps to manipulation (L2-L4), label induction or the ICL suite; the paper's own scope statement excludes reverse relations, so reverse merchant tasks would not benefit.
- LoRA interaction: not studied here (see 2511.05852).
- Qwen: not tested; layers R must be re-located for Qwen2.5-3B by causal tracing.
- Cost: dominated by per-edit z optimisation (about 2.4 s per edit on GPT-J serially) and one-time covariance collection over 100k samples in fp32; 680 edits on a 3B model is plausibly under an hour on a 3090, but this is an extrapolation, not a paper number.

## Key references worth following up
2202.05262 (ROME, causal tracing), 2110.11309 (MEND), 2206.06520 (SERAC), 2012.14913 (Geva et al., FFN as key-value memories), 2012.00363 (Zhu et al., constrained FT), 2104.08696 (knowledge neurons), 2111.13654 (Hase et al., beliefs).
