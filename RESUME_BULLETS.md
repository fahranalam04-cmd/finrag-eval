# Resume bullets: finrag-eval

**Financial RAG & LLM Evaluation** | Python · PyTorch · Hugging Face · Qdrant · FastAPI

- Built hybrid RAG over 57.6K FiQA financial passages (BM25 + bge-base in Qdrant, weighted RRF, cross-encoder fusion); raised test nDCG@10 from 0.250 to 0.421 (+68% vs dev-tuned BM25) on 648 queries [1]
- Wrote nDCG/Recall/MRR metrics matching pytrec_eval to 1e-15 plus paired bootstrap tests; showed default RRF hurt (-0.032 vs dense) and that only dev-tuned fusion and CE rank fusion helped (p<0.02) [2]
<!-- GEN-BULLET -->
- Served /search and /answer via FastAPI + Streamlit; profiled per-stage p50/p95 latency and traced 94% of hybrid p50 (385 of 411 ms) to embedded Qdrant vs 10.7 ms for exact NumPy search [4]

## Alternates

<!-- ALT -->

## Sources (every number above)

<!-- SOURCES -->
