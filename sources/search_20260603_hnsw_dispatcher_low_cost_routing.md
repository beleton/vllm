# HNSW dispatcher low-cost routing sources

Date: 2026-06-03

This file records external sources consulted for the CCD-aware HNSW dispatcher analysis.

## Sources

- QVCache: A Query-Aware Vector Cache, arXiv:2602.02057, 2026-02-02. https://arxiv.org/abs/2602.02057
- CCD-Level and Load-Aware Thread Orchestration for In-Memory Vector ANNS on Multi-Core CPUs, arXiv:2605.10090, 2026-05-11. https://arxiv.org/abs/2605.10090
- Graph Reordering for Cache-Efficient Near Neighbor Search, NeurIPS 2022. https://proceedings.neurips.cc/paper_files/paper/2022/hash/fb44a668c2d4bc984e9d6ca261262cbb-Abstract-Conference.html
- Graph Reordering for Cache-Efficient Near Neighbor Search, arXiv:2104.03221. https://arxiv.org/abs/2104.03221
- Enhancing Similarity Search Throughput by Dynamic Query Reordering, DEXA 2016. https://is.muni.cz/publication/1352385/cs
- Similarity Estimation Techniques from Rounding Algorithms, STOC 2002. https://dblp.org/rec/conf/stoc/Charikar02
- LSH Forest: Self-Tuning Indexes for Similarity Search, WWW 2005. https://www.cs.princeton.edu/courses/archive/spring06/cos592/bib/LSHForest-bawa05.pdf
- Faiss IndexIVF documentation. https://faiss.ai/cpp_api/struct/structfaiss_1_1IndexIVF.html
- Faiss additive/coarse quantizer documentation. https://github.com/facebookresearch/faiss/wiki/Additive-quantizers

## Notes

- External sources support the existence of query-level routing, coarse quantization, LSH-style signatures, query reordering, query-aware caching, and CCD-aware ANNS scheduling.
- None of the external sources alone proves that a microsecond-level HNSW query dispatcher is deployable for the current single-table, multi-CCD hnswlib setting.
- Current local Lab4 evidence is therefore treated as primary for dispatcher cost and path-overlap bounds.
