# What Pinterest's recommender work says about the transaction categoriser (2026-09-21)

Sources: the 26 Pinterest engineering posts and the Zepto post in `pinterest/` and `other/` (fetched by
`scripts/fetch_blog_posts.py`, summarised in `pinterest/summaries.md`), the labs topic pages and the publication list in
`pinterest/publications.md`. Three posts that no script route could fetch, the five labs featured posts (abstracts that link to
their papers) and one more (Pinner Progression) were saved by the owner from a browser on 2026-09-21 and filed beside the
rest; the temporal-leakage post is a preview only (member-only story). Section 9 covers what they added. The owner's questions: is this a graph problem like Pinterest's, could a random walk over transaction
nodes discover that user A's "Fluffy" is other users' "Pets", and is all of that achievable by a well-tuned embedding model
and a well-crafted prompt without the graph.

## 1. The mapping is exact, and Pinterest built it twice

Pixie's graph is "the graph between Pins and boards", which the post calls "highly unique as it's created from how people
describe and organize Pins". Ours is the graph between merchants (or transactions) and user categories, created from how
people file transactions. The correspondences:

| Pinterest | transaction categoriser |
|---|---|
| Pin (image + annotations) | transaction (payee string, amount, weekday, type, location) and its merchant |
| board (a user's named collection) | a user's category ("Fluffy", "Pets", "Groceries") |
| a Pinner saving a Pin to a board | a user filing a transaction under a category |
| the same image saved to many boards | the same merchant filed under many users' categories |
| Pin content embeddings (visual, annotation) | the text embedding of the transaction string plus the merchant's fact-DB record |
| board "diverse" pruning (boards with Pins from many ideas) | dropping catch-all categories ("Misc") from the walk |
| popular-Pin neighbour caps | capping Amazon, Walmart, Venmo |
| random walk with restart from a query Pin | random walk with restart from the merchant of the incoming transaction |

The "Evolution of Search" post states the principle in one sentence: "people are taking the same image and putting it on
different boards, we can learn the deeper semantics of an image. We can then train our systems to emulate the ways Pinners
are categorizing images." Replace image with merchant and boards with categories and it is the owner's paragraph about
"Fluffy" and "PetSmart". PinSage's motivating example is our problem too: "a bed rail Pin might look like a garden fence,
but gates and beds are rarely adjacent in the graph"; a payee string can look like one thing and be another, and the users
who filed it tell you which.

What the walk yields, in our terms:

- One hop (merchant, its categories across users): the distribution of category names under which other users filed
  PetSmart, which is a collaborative record beside the fact-DB's content record ("sells pet food, toys and grooming").
- Two hops (category, its merchants, their categories): categories that share merchants are synonyms, so "Fluffy" lands
  next to "Pets" and "Dog stuff" because they contain PetSmart, Chewy and the vet; and the incoming transaction's nearest
  categories in the user's own scheme follow from that. This is the synonym-category discovery the owner described, and it
  is exactly what Pixie's visit counts produce.

Pinterest built this first as a pure walk (Pixie, 2017 to 2018: 100,000 steps, restart probability 0.5, visit counts as
relevance, "37x" engagement over popularity recycling) and then as a GNN whose neighbourhoods the walk defines (PinSage,
2018: "the neighbors with the highest visit counts", "a 46% performance gain over the traditional K-hop graph
neighborhood"). Both still run: Pixie is "one major source" of home-feed candidates in the 2021 two-tower post, and XPixie
supplies the neighbours that LinkSage (2024) reads.

## 2. Graph or embeddings is the wrong dichotomy: the graph is the retrieval index, the model is the aggregator

The owner's second question is whether a good enough embedding model with a well-crafted prompt makes the graph
unnecessary. Pinterest's own evolution answers it, and so does what the graph does at the model level:

- PinSage is not an alternative to embeddings; it is the embedding model, trained so that a node's vector is an aggregate of
  its own content features and its neighbours' content features, where the neighbours are chosen by the random walk. The
  graph is the training signal and the neighbourhood sampler; the network does the rest.
- LinkSage makes the structure explicit: a transformer reads the node's own text and image features next to its neighbours'
  features, "reverse sorted by the visited counts", plus visit counts and degree as tokens. That is a prompt with retrieved
  context. Our record-in-prompt categoriser is a one-layer GNN whose neighbourhood is chosen by retrieval: the user's own 24
  rows, and the merchant's record. It reads them with a 3B transformer instead of a small aggregator.

So the real question is not graph versus embeddings but which neighbours the model gets to see, and whether the cross-user
signal reaches it parametrically or non-parametrically:

1. The user's own history: in the prompt already.
2. The merchant's content: the fact-DB record, in the prompt already; this is Pinterest's cold-start answer too (content
   features, semantic IDs from content in the 2026 retrieval post, "Warmer for Less" in the publication list).
3. What other users call the category this merchant goes into, and which categories are synonyms: not in the prompt. The
   label SFT over 20 users' histories puts a version of it in the weights (it is where the unseen-merchant cells gain over
   the base), and a fine-tune over a million users' histories would put more of it there. Parametric has two costs
   Pinterest names: drift (the realtime-actions ranker "decayed unless retrained", so it is retrained twice a week; the
   Closeup post found "model refreshes bring in better performance across all refresh cadence"), and a new merchant's first
   labels are invisible until the next retrain. Non-parametric is a lookup: the merchant's category histogram and its
   two-hop synonyms, rendered as a line in the prompt, updated the moment a label arrives.

Section 43 of the report already ran this contrast for the content record: in the weights, +10 +- 12 on DB-only merchants
across seeds; in the prompt, +42 with a spread of 6. The same experiment for the collaborative record is the missing row,
and the prior from both Pinterest and our own results is that the prompt wins.

The honest limit of "a good enough model learns it from the prompt": it can only learn from what is in the prompt, and the
cross-user structure is not in any single user's prompt. It has to be put there by a retrieval step (the graph, or an
equivalent lookup) or baked into the weights by training on many users. There is no third way, and the graph is the
cheaper, fresher of the two.

## 3. Cold start and per-user modification

The owner's point that prompting a fine-tuned model avoids both the cold-start problem and the per-user-model problem is
what Pinterest's two-tower post says about user embeddings: Pin embeddings are precomputed, but "we enable online user
embedding computation whenever there is a new user request" because "Pinners tend to change their status instantly". One
shared model, the user's state recomputed from their latest history on every request. Two refinements from the posts:

- A brand-new user has no history, and the content record alone does not say which category most people use for PetSmart.
  The collaborative record does. Cold-start of a user is therefore the collaborative record's job, cold-start of a merchant
  the content record's.
- Pinterest kept a heavy ranker behind the two-tower stage because a dot product "cannot cross" user and item features. Our
  centroid route is the two-tower model and the LLM prompt is the cross-attention ranker; that is why the prompt beats the
  centroid (90 vs 84) and why the cascade (cheap model for confident cases, LLM for the rest) is the cost design, not a
  replacement.

## 4. The real-time lesson is about which history rows the model sees

The largest measured gains in the whole set come from changing which of the user's actions the ranker sees, not from model
size:

- TransAct (2022): the last 100 actions as a sequence with early fusion against the candidate: "+6% repin volume overall,
  +11% for non-core (new, casual, resurrected) users", the biggest lift on low-history users.
- TransAct V2 (2025): 16,000 lifelong actions, of which each candidate sees "the most recent r actions" plus "the top K
  nearest neighbors ... based on dot-product similarity", selected per candidate: "+13.31% Top-3 Repin Hit ... more than 2x
  improvement over previous systems" where "previous model launches ... typically achieved lifts around 0.2-1%".
- Zepto (2026): separate encoders for long history and the current session, target-aware pooling ("the user profile is
  dynamically recomputed per candidate"), a learned gate between them, elapsed hours as a feature per interaction.

For us this is PLAN row 41 (REAL-9) with its design already chosen: the shots are the most recent r plus the K nearest to the
candidate, per candidate, combined; not similarity or recency but both. Two further transfers: mark each shot with how the
label arose (the user typed it, accepted a suggestion, or corrected one), which is TransAct's action type; and add the
elapsed time per shot ("3 days ago"), which is Zepto's temporal encoding. TransAct V2's finding that impression negatives
beat random negatives maps to training with the categories the user saw and did not pick as the negatives, which the label
SFT does implicitly and the encoder route should do explicitly.

## 5. The encoder route, as Pinterest would build it

SearchSage and the two-tower post are a recipe: freeze one side (the transaction encoder), fine-tune the other end to end
(a small text encoder for category names into the same space), in-batch softmax at a large batch (6,000 at Pinterest),
cap how often any one item appears as a positive so head categories do not dominate, use only user-confirmed labels as
positives, and, from the deduplication post, never draw a negative from the same user. PinSage adds the merchant-side
recipe: the merchant's embedding aggregates its record text with its walk neighbours' features and visit counts. The
retrieval-platform post's late interaction (Sum of MaxSim over several vectors per side) is FastFit's mechanism, already row
40.

## 6. Evaluation and serving notes

- Point-in-time correctness: the user-sequence platform posts and the Closeup post insist that the training path and the
  serving path build the sequence with one definition, and that a small randomised slice is logged for unbiased offline
  replay. Row 43's set should have both: shots built by the same code the scorer uses, and a slice whose shots are drawn
  uniformly so a shot-selection policy can be replayed.
- Scoring cost: the deduplication post's DCAT and the CLR post's M-Falcon encode the user's history once and let every
  candidate attend to it. For option scoring that is the shared-prefix KV cache: one forward for the 24 shots and record,
  one short continuation per category name. The batched Scorer approximates it; the exact form is the engineering item.
- Scale: Pixie's pruned graph was 20 billion edges in 150 GB of RAM. A million users with a few thousand transactions
  each and tens of thousands of merchants is a graph of a few billion edges at most, and a merchant-category bipartite
  graph (merchants against standardised category nodes per user) is far smaller. The walk itself is not the hard part.

## 7. What does not transfer

The serving and training infrastructure (Ray, MLEnv, Flink, near-linear multi-node scaling, feature trimming, ANN index
quantisation), multi-objective blending (MMoE, Learned Utility), the ads and churn models, content moderation, and the
16,000-token lifelong sequences as a model input: our prompt holds about 24 rows, which is why retrieval over the history,
not a longer context, is the transfer.

## 8. Proposed additions to PLAN.md

1. **Row 44, the collaborative record (Pixie in one and two hops).** For each merchant, the histogram of the category names
   other users filed it under, mapped to standard names (one hop), and the categories reached by a random walk with restart
   over the bipartite merchant-category graph of the training users (two hops, Pixie's rule: visit counts, restart 0.5,
   catch-all categories and hub merchants pruned), rendered as a second note line in the prompt: "Other users file this
   merchant under: Pets 61%, Shopping 20%". Arms: no record, content record, collaborative record, both; the untrained base
   and the SFT categoriser; REAL-6 now (other users' labels exist for every merchant that is not DB-only) and row 43's set
   with idiosyncratic assignments once it exists, where the graph is the only source of a shared personal reason.
2. **Row 41 (REAL-9) revised** to TransAct V2's rule as the third arm: the most recent r plus the K nearest per candidate.
3. **Row 43 (REAL-11) extended** with an action type per shot (typed, accepted, corrected), elapsed time per shot, and a
   uniformly-drawn-shots slice for replaying selection policies.
4. **The encoder route (row 40 or later)** with the SearchSage recipe and a PinSage-style merchant embedding from record plus
   walk neighbours.

## 9. Addendum from the hand-saved posts (2026-09-21, later the same day)

- **Point-in-time correctness (the temporal-leakage preview).** The rule as stated: "a training row built for an event at
  time T must reflect only what was knowable at or before T", and the opening example is our record-in-prompt route with
  the wrong index: a copilot retrieving a five-month-old "low risk" note by similarity while the point-in-time fraud model
  reads the last ten minutes. Row 43's set must filter every retrieved shot, record and collaborative record by the
  transaction's timestamp, at training and at evaluation. Added to REAL-11 as a requirement rather than a note.
- **Per-user clusters as the shot-selection unit (Pinner Progression).** Cluster the user's own labelled transactions in
  their own space, medoid plus landmarks per category, recency and frequency per cluster, a dynamic cluster count (two for a
  new user, fifteen for a heavy one), and shots chosen by cluster coverage with a same-cluster discount and frontier
  sampling at the boundaries. This is a fourth shot rule for row 41 beside similarity, recency and TransAct V2's
  combination, and the 0.85-cosine assignment threshold with a default group is an abstain rule for the encoder route.
- **Auto-apply feedback loops (multi-objective optimisation).** A gain on day one that turns negative by week two is
  Pinterest's experience of letting the system's own choices feed its signals. Auto-applied categories entering the shot
  history are the same loop; judge auto-apply on later correction rates, and prefer a graded confidence penalty to a hard
  ask threshold. Belongs in row 43's evaluation (a correction-rate measure after auto-applied labels enter the history).
- **Payee-string canonicalisation (MIQPS).** A token is noise if removing it leaves the label unchanged, decided per merchant
  and per string pattern with conservative defaults and a guard against refreshes that silently drop a token. A design
  note for the retriever's normaliser, not a row.
- **The labs abstracts (PinnerFormer, OmniSearchSage, OmniSage, PinFM, UniPinRec)** confirm the reading above: one
  embedding trained from graph, content and sequence signals together (OmniSage); a daily batch user summary trained to
  predict the next two weeks of actions, closing most of the gap to realtime embeddings (PinnerFormer); entity text enriched
  by an LLM and by "user-curated boards" (OmniSearchSage), which is the content record plus the collaborative record. The
  papers are the next reading if the encoder route is pursued at scale.
