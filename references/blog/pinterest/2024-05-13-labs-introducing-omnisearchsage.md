# Introducing OmniSearchSage: Multi-Task Multi-Entity Embeddings for Pinterest Search

- Source: https://labs.pinterest.com/article/introducing-omnisearchsage
- Linked from: https://labs.pinterest.com/research-and-innovation/representation-learning (Featured Posts)
- Publisher: Pinterest Labs
- Published: 2024-05-13
- Authors: Prabhat Agarwal, Minhazul Islam Sk, Nikil Pancha, Kurchi Subhra Hazra, Jiajing Xu, Chuck Rosenberg (WWW 2024)
- Topic: Representation Learning
- Paper: ACM https://dl.acm.org/doi/10.1145/3589335.3648309 | arXiv https://arxiv.org/abs/2404.16260
- Captured: 2026-09-21 via Chrome page text extraction.

**Note:** The Pinterest Labs "featured post" page is an abstract that links out to the paper. There is no longer-form blog text on this page. The full content is the arXiv paper above.

---

OmniSearchSage, a versatile and scalable system for understanding search queries, pins, and products for Pinterest search. We jointly learn a unified query embedding coupled with pin and product embeddings, leading to an improvement of > 8% relevance, > 7% engagement, and > 5% ads CTR in Pinterest's production search system.

The main contributors to these gains are improved content understanding, better multi-task learning, and real-time serving. We enrich our entity representations using diverse text derived from image captions from a generative LLM, historical engagement, and user-curated boards. Our multitask learning setup produces a single search query embedding in the same space as pin and product embeddings and compatible with pre-existing pin and product embeddings. We show the value of each feature through ablation studies, and show the effectiveness of a unified model compared to standalone counterparts. Finally, we share how these embeddings have been deployed across the Pinterest search stack, from retrieval to ranking, scaling to serve 300k requests per second at low latency.
