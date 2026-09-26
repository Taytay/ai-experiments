# Introducing PinFM: Foundation Model for User Activity Sequences at Pinterest

- Source: https://labs.pinterest.com/article/introducing-pinfm
- Linked from: https://labs.pinterest.com/research-and-innovation/recommender-systems (Featured Posts)
- Publisher: Pinterest Labs
- Published: 2025-09-22
- Authors: Xiangyi Chen, Kousik Rajesh, Matthew Lawhon, Zelun Wang, Hanyu Li, Haomiao Li, Saurabh Vishwas Joshi, Pong Eksombatchai, Jaewon Yang, Yi-Ping Hsu, Jiajing Xu, Charles Rosenberg (RecSys 2025)
- Topic: Recommender Systems
- Paper: ACM https://dl.acm.org/doi/full/10.1145/3705328.3748050 | arXiv https://arxiv.org/abs/2507.12704
- Captured: 2026-09-21 via Chrome page text extraction.

**Note:** The Pinterest Labs "featured post" page is a one-paragraph abstract that links out to the paper. There is no longer-form blog text on this page. The full content is the arXiv paper above.

---

PinFM is a large, 20B-parameter sequential model pretrained on lifelong user activity sequences. It plugs into downstream ranking systems (e.g., Homefeed) and is fine-tuned per use case. The result is a step-function improvement in user-sequence understanding without sacrificing the ability to leverage existing features and components.
