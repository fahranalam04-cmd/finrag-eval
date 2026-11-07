"""Retrieve-then-generate pipeline used by the API, and a loader for the tuned on-disk artefacts."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from finrag.config import EMBEDDERS, GENERATOR_HF_ID, RERANKERS, Paths
from finrag.generation import ContextPassage, Generator
from finrag.retriever import Hit, HybridRetriever, Mode, RetrieverConfig, SearchResult


@dataclass
class Citation:
    doc_id: str
    in_context: bool


@dataclass
class Answer:
    query: str
    answer: str
    abstained: bool
    citations: list[Citation]
    passages: list[Hit]
    timings_ms: dict[str, float] = field(default_factory=dict)


class RAGPipeline:
    def __init__(self, retriever: HybridRetriever, generator: Generator | None = None) -> None:
        self.retriever, self.generator = retriever, generator

    @property
    def default_mode(self) -> Mode:
        """The dev-selected pipeline: rerank only if a reranker was selected (and loaded)."""
        return "hybrid_rerank" if self.retriever.reranker is not None else "hybrid"

    def search(self, query: str, mode: Mode | None = None, k: int = 10) -> SearchResult:
        return self.retriever.search(query, mode or self.default_mode, k)

    def answer(self, query: str, mode: Mode | None = None, k_context: int = 5) -> Answer:
        if self.generator is None:
            raise RuntimeError("no generator configured")
        res = self.retriever.search(query, mode or self.default_mode, k_context)
        t0 = time.perf_counter()
        context = [ContextPassage(h.doc_id, h.text) for h in res.hits]
        gen = self.generator.generate_batch([(query, context)])[0]
        timings = dict(res.timings_ms)
        timings["generate"] = (time.perf_counter() - t0) * 1e3
        timings["total"] = timings.get("total", 0.0) + timings["generate"]
        ctx = {h.doc_id for h in res.hits}
        return Answer(
            query=query,
            answer=gen.answer,
            abstained=gen.abstained,
            citations=[Citation(d, d in ctx) for d in gen.cited_ids],
            passages=res.hits,
            timings_ms=timings,
        )


def tuned_settings(paths: Paths | None = None) -> tuple[str, RetrieverConfig, str | None]:
    """Dev-selected embedder / fusion / reranker from reports/dev_tuning.json (defaults if absent)."""
    try:
        from finrag.experiments.retrieval import selected_config

        return selected_config()
    except (FileNotFoundError, KeyError):
        return "bge-small", RetrieverConfig(), None


def load_retriever(
    paths: Paths, device: str, with_reranker: bool = True, reranker_name: str | None = None
) -> HybridRetriever:
    """Load the dev-tuned retriever. The reranker is the dev-selected one, unless `reranker_name`
    forces one (used to time reranking even when it was not selected)."""
    from finrag.bm25 import BM25Index
    from finrag.dense import QdrantDenseIndex
    from finrag.embedders import SentenceTransformerEmbedder
    from finrag.experiments.index import QDRANT_COLLECTION
    from finrag.rerank import CrossEncoderReranker

    emb_name, cfg, rr_name = tuned_settings(paths)
    rr_name = reranker_name or rr_name
    e = EMBEDDERS[emb_name]
    embedder = SentenceTransformerEmbedder(
        e["hf_id"], emb_name, e["query_prefix"], e["passage_prefix"], device=device, batch_size=1
    )
    reranker = (
        CrossEncoderReranker(RERANKERS[rr_name], rr_name, device=device, batch_size=cfg.rerank_depth)
        if with_reranker and rr_name
        else None
    )
    return HybridRetriever(
        BM25Index.load(paths.bm25_dir),
        QdrantDenseIndex.local(paths.qdrant_dir / emb_name, QDRANT_COLLECTION),
        embedder,
        reranker,
        config=cfg,
    )


def load_pipeline(paths: Paths, device: str, with_generator: bool = True) -> RAGPipeline:
    generator = None
    if with_generator:
        from finrag.generation import HFLocalGenerator

        generator = HFLocalGenerator(GENERATOR_HF_ID, device=device, batch_size=1)
    return RAGPipeline(load_retriever(paths, device), generator)
