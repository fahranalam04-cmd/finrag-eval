# Resume bullets: finrag-eval

**Financial RAG & LLM Evaluation** | Python · PyTorch · Hugging Face · Qdrant · FastAPI

- Built hybrid RAG over 57.6K FiQA financial passages (BM25 + bge-base in Qdrant, weighted RRF, cross-encoder fusion); raised test nDCG@10 from 0.250 to 0.421 (+68% vs dev-tuned BM25) on 648 queries [1]
- Wrote nDCG/Recall/MRR metrics matching pytrec_eval to 1e-15 plus paired bootstrap tests; showed default RRF hurt (-0.032 vs dense) and that only dev-tuned fusion and CE rank fusion helped (p<0.02) [2]
- Evaluated Qwen2.5-1.5B on 150 test queries with NLI faithfulness and qrels citation checks: 97% of cited ids valid, citation precision 0.36 vs 0.25 base rate; traced low faithfulness to missing citations [3]
- Served /search and /answer via FastAPI + Streamlit; profiled per-stage p50/p95 latency and traced 94% of hybrid p50 (385 of 411 ms) to embedded Qdrant vs 10.7 ms for exact NumPy search [4]

## Alternates

- Built a tune-on-dev, test-once harness with a leakage guard, bootstrap CIs and an ir_measures cross-check (max diff 4.4e-16) across 9 retrieval variants on 648 FiQA test queries [2]
- Showed textbook RRF (k=60) cut nDCG@10 7.8% vs dense-only, while dev-tuned weighted RRF plus cross-encoder rank fusion raised it 3.8% (0.406 to 0.421), each tested with paired bootstrap [2]
- Profiled per-stage latency: hybrid + rerank ran 460 ms p50 on GPU vs 10.8 s on CPU (96% in reranking), so CPU deployments serve the tuned hybrid at 0.413 nDCG@10 instead [4]

## Sources (every number above)

All results are on the FiQA-2018 test split, scored once with settings frozen on dev.

1. `reports/data_stats.json`: `n_passages` = 57638.
   `reports/retrieval_test.json`:
   - `n_queries` = 648
   - `variants.bm25.ndcg@10` = 0.2502
   - `variants["hybrid+rerank:bge-reranker-base"].ndcg@10` = 0.4213
   - `paired_bootstrap_ndcg@10["hybrid+rerank:bge-reranker-base vs bm25"].relative_lift` = 0.6839
2. `reports/retrieval_test.json`:
   - `variants.*.crosscheck.max_abs_diff`: at most 4.4e-16 over 9 variants.
   - `paired_bootstrap_ndcg@10`:
     - `["hybrid:rrf60 vs dense:bge-base"]`: diff = -0.0316, relative_lift = -0.0778
     - `["hybrid:tuned vs dense:bge-base"]`: p_value = 0.015
     - `["hybrid+rerank:bge-reranker-base vs hybrid:tuned"]`: p_value = 0.0123
   - `variants["dense:bge-base"].ndcg@10` = 0.4058 vs 0.4213 for the selected variant, which is +3.8%.
3. `reports/generation_eval.json`:
   - `n_queries_per_config.selected_top3` = 150
   - `configs.selected_top3`:
     - `citation_validity` = 0.9747
     - `citation_precision_qrels` = 0.3636
     - `context_precision_qrels` = 0.2467 (base rate)
     - `faithfulness_sentence` = 0.232
     - `context_faithfulness_sentence` = 0.7643, the evidence that the gap is missing citations
4. `reports/latency.json`:
   - `cuda.modes.hybrid.stages_ms.dense_search.p50` = 385.18 and `cuda.modes.hybrid.stages_ms.total.p50` = 410.6
   - `numpy_exact_dense_reference.ms.p50` = 10.73
   - `cuda.modes.hybrid_rerank.stages_ms.total.p50` = 459.9
   - `cpu.modes.hybrid_rerank.stages_ms.total.p50` = 10774.41 and `cpu.modes.hybrid_rerank.stages_ms.rerank.p50` = 10391.82
   - `variants["hybrid:tuned"].ndcg@10` = 0.4128 comes from `retrieval_test.json`.
