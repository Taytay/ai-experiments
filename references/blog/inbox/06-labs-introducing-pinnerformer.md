# Introducing PinnerFormer: Sequence Modeling for User Representation at Pinterest

- Source: https://labs.pinterest.com/article/introducing-pinnerformer
- Linked from: https://labs.pinterest.com/research-and-innovation/representation-learning (Featured Posts)
- Publisher: Pinterest Labs
- Published: 2022-08-14
- Authors: Nikil Pancha, Andrew Zhai, Jure Leskovec, Charles Rosenberg (KDD 2022)
- Topic: Representation Learning
- Paper: ACM https://dl.acm.org/doi/10.1145/3534678.3539156 | arXiv https://arxiv.org/abs/2205.04507
- Captured: 2026-09-21 via Chrome page text extraction.

**Note:** The Pinterest Labs "featured post" page is an abstract that links out to the paper. There is no longer-form blog text on this page. The full content is the arXiv paper above.

---

Sequential models have become increasingly popular in powering personalized recommendation systems over the past several years. These approaches traditionally model a user's actions on a website as a sequence to predict the user's next action. While theoretically simplistic, these models are quite challenging to deploy in production, commonly requiring streaming infrastructure to reflect the latest user activity and potentially managing mutable data for encoding a user's hidden state.

Here we introduce PinnerFormer, a user representation trained to predict a user's future long-term engagement using a sequential model of a user's recent actions. Unlike prior approaches, we adapt our modeling to a batch infrastructure via our new dense all-action loss, modeling long-term future actions instead of next action prediction. We show that by doing so, we significantly close the gap between batch user embeddings that are generated once a day and realtime user embeddings generated whenever a user takes an action. We describe our design decisions via extensive offline experimentation and ablations and validate the efficacy of our approach in A/B experiments showing substantial improvements in Pinterest's user retention and engagement when comparing PinnerFormer against our previous user representation. PinnerFormer is deployed in production as of Fall 2021.
