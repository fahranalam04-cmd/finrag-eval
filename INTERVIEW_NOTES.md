# Interview notes: finrag-eval

Draft. Results-dependent sections are filled in once the test run is frozen.

## Design choices and why

**FiQA-2018 (BEIR) as the benchmark.** It is a real financial-domain retrieval task with public qrels
and published baselines (BEIR BM25 nDCG@10 = 0.236), so every number here can be sanity-checked
against the literature. Its weakness: the passages are StackExchange/Reddit-style opinion answers, not
filings, and qrels are sparse (about 2.6 judged passages per query), which matters for how
citation precision should be read (see weaknesses).

**Protocol: tune on dev, report test once.** Everything that is a choice (BM25 k1/b, embedder, RRF k
and dense weight, reranker model and depth, the generation prompt) is selected on the 500 dev
queries. `experiments.common.require_tuning_split` raises if any tuning function is pointed at test,
and `evaluate_split` reads only the frozen selections in `reports/dev_tuning.json`.

**Tuned BM25 baseline, not a strawman.** The lift claims are against BM25 with dev-tuned k1/b and
Snowball stemming, which is a stronger baseline than the BEIR Elasticsearch default.

**Qdrant in embedded mode.** No server, no Docker, persisted under D:. Embedded mode does exact
(brute-force) cosine search, and qdrant-client itself warns that HNSW search params are ignored, so
the dense numbers carry no ANN approximation error. The collection is created with an HNSW config
so the same code (`QdrantDenseIndex.remote`) uses HNSW when pointed at a server. At 57.6K vectors
exact search is a few milliseconds; I would switch to a server with HNSW past a few million.

**RRF instead of score interpolation.** BM25 scores and cosine similarities live on different scales
and their distributions shift per query; RRF only needs ranks, so it needs no normalisation and has
one or two knobs (k, a per-list weight), both tuned on dev.

**Rerank only the head.** A cross-encoder costs one transformer forward pass per (query, passage)
pair, so it is applied to the top-N fused candidates only; the tail keeps its fused order. N is
tuned on dev. Cross-encoder scores are computed once for the deepest N in the grid and reused, so the
depth sweep costs nothing extra.

**Generator behind a protocol.** `Generator.generate_batch(items) -> list[GenerationResult]` is the
whole interface; the local Qwen2.5-1.5B-Instruct backend and the test fake both implement it, and an
API or vLLM backend would only need that one method.

**Faithfulness via NLI, not an LLM judge.** A DeBERTa-v3 NLI model is cheap, deterministic, local and
has known behaviour; an LLM judge would need a bigger model or an API key. The metric is sentence
level: each answer sentence must be entailed by a passage it cites.
