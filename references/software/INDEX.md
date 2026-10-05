# Software references

Documentation and key source collected from open-source libraries we use or evaluate (text only; each folder has COMMIT = the upstream
commit it was taken from, and LICENSE).

## Tutte Institute: embedding clustering, topic naming, maps (collected 2026-10-05 for PLAN row 213)
- [evoc/](evoc/) — EVōC, Embedding Vector Oriented Clustering (Leland McInnes; BSD-2-Clause; v0.3.x, Feb-Mar 2026, early beta). Clusters
  embedding vectors fast on CPU: a UMAP-like node embedding of a kNN graph, HDBSCAN-style density clustering over a minimum spanning tree,
  and persistence across several granularities. `cluster_layers_` (finest first), `cluster_tree_` (the hierarchy between layers),
  `duplicates_` (near-duplicate vectors), automatic cluster count, int8/binary vectors. No paper: README.rst, doc/ (user guide,
  quickstart, examples, API index, changelog), doc/benchmarks.md (the benchmark notebook's text and printed outputs: claims better than
  UMAP + HDBSCAN, faster than k-means), and the core source (clustering.py, clustering_utilities.py).
- [toponymy/](toponymy/) — Toponymy (MIT): names clusters with an LLM, layer by layer, into a TopicTree; treemap.py's
  `treemap_dataframe(topic_tree)` (moved into the topic tree in Feb 2026) turns that tree into a plotly treemap: the "fast treemap of
  related topics" the owner remembered.
- [datamapplot/](datamapplot/) — DataMapPlot (MIT): static and interactive maps of 2-D embeddings with layered labels.

Uses here: row 213 (EVōC in place of row 189's k-means 64 category clusters); later row 160 (payee resolution via `duplicates_`) and a
category treemap for exploration.
