# Introducing UniPinRec: Unifying Generative Retrieval and Ranking at Pinterest Scale

- Source: https://labs.pinterest.com/article/introducing-unipinrec
- Linked from: https://labs.pinterest.com/research-and-innovation/recommender-systems (Featured Posts)
- Publisher: Pinterest Labs
- Published: 2026-05-29
- Authors: Hanyu Li, Yi-Ping Hsu, Aditya Mantha, Prabhat Agarwal, Laksh Bhasin, Jialu Wang, Hongtao Lin, Bella Huang, Yaxin Li, Xinyi Li, Chuxi Wang, Kousik Rajesh, Hooshmand Shokri Razaghi, Shunyao Li, Zongyue Qin, Jaewon Yang, James Li, Dhruvil Deven Badani, Jiajing Xu, Charles Rosenberg (preprint 2026)
- Topic: Recommender Systems
- Paper: arXiv https://arxiv.org/abs/2606.00422
- Captured: 2026-09-21 via Chrome page text extraction.

**Note:** The Pinterest Labs "featured post" page is a one-paragraph abstract that links out to the paper. There is no longer-form blog text on this page. The full content is the arXiv paper above.

---

Modern recommendation systems predominantly train retrieval and ranking as separate models despite both increasingly relying on large transformers encoding the same user behavior data, duplicating parameters, compute, and serving cost. Prior work unifies the model architecture but not the full pipeline: input formats, training procedures, and serving stacks remain fragmented across stages. We present UniPinRec, which achieves full-stack unification of retrieval and ranking at Pinterest: one input format, one model, one training stage, deployed within existing serving infrastructure.
