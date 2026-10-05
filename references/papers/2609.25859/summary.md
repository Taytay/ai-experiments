# BELXTR: Biomedical Entity Linking via Contextualized Token Retrieval

- arXiv 2609.25859 (v1, 2026-09-22) - https://arxiv.org/abs/2609.25859
- Samuele Garda, Ulf Leser (Humboldt-Universität zu Berlin)
- Venue: preprint (OUP template, journal fields blank; probably a Bioinformatics submission, inferred)
- Source: references/papers/2609.25859/paper_flat.tex (read 2026-10-05 for PLAN rows 209-210)

## One-paragraph summary
BELXTR adapts XTR (late-interaction token retrieval) to biomedical entity linking: a mention encoded with its context is scored against KB entity names by sum-of-max over token similarities, training counting only token pairs inside the in-batch top-k. Three changes to XTR: each entity *name* is its own candidate ("name-based"), trained with a maximum-marginal-likelihood (MML) loss summing probability over all names of the gold entity; a learnable CLIP-style temperature on the MaxSim score; optional supervised [MASK] query expansion. Best on 5 of 10 BELB corpora, second on 4; gains come mostly from gene corpora (NLM-Gene 82.29 vs 66.43). The dev ablation is the useful part: no temperature -9 to -15 recall@1; concatenating an entity's names into one document -2.8 to -21.2; [MASK] expansion about neutral.

## Problem
Single-vector bi-encoders average away fine surface differences ("α2-" vs "β2-microglobulin"; 33 ADAM genes differing by 1-2 characters). About 70% of NLM-Gene mentions have top-5 false-positive names at similarity >= 80 to the gold name.

## Method
- Mention encoded with as much context as the backbone allows; only mention-token vectors kept as Q (m x h); each candidate name gives D (n x h). Backbone, hidden size and projection are not named in the TeX (code: github.com/sg-wbi/belxtr).
- Test-time score f(Q,D) = (1/n) sum_i max_j Q_i.D_j (cosine). Training: XTR alignment, a token pair counts only if token j is in the top-k of the mini-batch tokens.
- Entity-based arm: an entity = concatenation of all its names, plain CE over candidates. Name-based: each name a candidate, p(D_i|Q) = softmax of f over candidates, L_MML = -log sum_i 1[entity(Q)=entity(D_i)] p(D_i|Q) (from BioSyn). Homonyms get a disambiguating parenthetical; gene names get species appended.
- Learnable temperature tau on every MaxSim score (CLIP), since cosines in [-1,1] give vanishing softmax gradients otherwise.
- Query expansion: [MASK] tokens appended (count = gold-name tokens missing from the mention), multiple-instance loss plus a dispersion loss, lambda 0.3; used only for gene and cell line.
- Training: <= 5 epochs, 4 queries per batch, 32 mined hard negatives per query, all in-batch negatives shared (<= 128 per query), lr 3e-6; top-k 2048 (the largest tried) best; tuned on NCBI Disease dev and reused.

## Experiments and results
- Dev ablation (recall@1, NCBI Disease / BC5CDR-chem / NLM-Gene): full 90.34 / 96.22 / 90.66; entity-based -4.92 / -2.78 / -21.22; no temperature -12.95 / -9.45 / -15.57; [MASK] without QE loss -0.89 / -1.01 / -0.63; with QE loss -0.48 / -0.17 / +0.58.
- Test recall@1 vs best baseline: NCBI Disease 88.75 vs 87.60; BC5CDR-disease 89.87 vs 89.23; BC5CDR-chem 95.46 vs 95.10; NLM-Chem 80.56 vs 82.39; BioID 95.72 vs 96.99; GNormPlus 84.82 vs 77.84; NLM-Gene 82.29 vs 66.43; S800 88.14 vs 89.96; Linnaeus 82.03 vs 88.60; MedMentions 69.47 vs 70.58.
- As retriever for GPT-4o reranking: standalone BELXTR 78.65 on refined NLM-Gene vs BioSyn+GRF+GPT-4o 69.17; BELXTR+GPT-4o drops to 75.96 (the reranker saw only the sentence, lacking species). On disease/chemical, GRF candidates rerank better (BELXTR optimises top-1, not recall).
- Out of corpus (BioRED, F1): gene 83.86 vs PubTator3 84.63; disease 83.34 vs BELHD 84.32; chemical 81.50 vs 83.93 (in-corpus win reverses).
- Efficiency (NLM-Gene, RTX 3090): single-vector 57.93 recall@1, 15.94 abstracts/s, 0.75 GB; multi-vector 82.29, 2.38/s, 5.90 GB.
- Advantage largest on near-identical candidate sets.

## Limitations
Baselines not re-tuned; XTR's top-k objective never ablated against plain sum-of-max (with k=2048 over 4 queries it is probably unrestricted, inferred); backbone unnamed; single runs; no calibration analysis; multi-vector slower and larger.

## Relevance to this workspace
- Learnable temperature on the MaxSim score is the largest single effect (9-15 points): row 210 softmaxes MaxSim over the household's options and has one (li_decider.py `scale`, init 20, lr 1e-2).
- Name-based + MML vs entity-based concatenation maps onto row 210's design: one document per category made of its filings is the "entity-based" arm that lost 2.8-21 points, worst where names are heterogeneous (household categories hold heterogeneous payees). Arm: each earlier filing (plus the "Group: Name" text) its own candidate, softmax over all, MML over the gold category's candidates, P(category) = summed mass.
- Homonyms: the same payee filed under different categories is common; append what disambiguates (amount, weekday) to each filing's text.
- A decider fed candidates from the retriever needs the same history evidence the retriever had (GPT-4o lost 2.7 points without it).
- Judge a candidate generator by recall@k, not top-1. Expect multi-vector gains only where candidates differ by surface tokens; our categories differ by purpose, so row 210's gain has to come from the decision objective (softmax over options, MML, temperature), not token interaction alone.
- Nothing transfers to row 209 (no base comparison).
