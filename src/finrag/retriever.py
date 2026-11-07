"""Online retrieval: BM25 | dense | hybrid (RRF) | hybrid + cross-encoder rerank, with stage timings.

The offline evaluator (finrag.experiments.retrieval) composes the same primitives in batch form;
tests/test_retriever.py checks that both paths return identical rankings.
"""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Literal, Protocol

from finrag.bm25 import BM25Index
from finrag.data import Passage
from finrag.dense import QdrantDenseIndex
from finrag.embedders import Embedder
from finrag.fusion import rrf
from finrag.rerank import Reranker, apply_rerank
from finrag.types import Ranked, ranked_ids

Mode = Literal["bm25", "dense", "hybrid", "hybrid_rerank"]
MODES: tuple[Mode, ...] = ("bm25", "dense", "hybrid", "hybrid_rerank")


class DocStore(Protocol):
    def fetch(self, doc_ids: Sequence[str]) -> Mapping[str, Passage]: ...


class DictDocStore:
    def __init__(self, passages: Sequence[Passage]) -> None:
        self._by_id = {p.doc_id: p for p in passages}

    def fetch(self, doc_ids: Sequence[str]) -> dict[str, Passage]:
        return {d: self._by_id[d] for d in doc_ids if d in self._by_id}


@dataclass(frozen=True)
class RetrieverConfig:
    candidates: int = 100  # depth retrieved from each first-stage retriever
    rrf_k: int = 60
    dense_weight: float = 1.0  # RRF weight of the dense list (BM25 weight fixed at 1)
    rerank_depth: int = 50
    rerank_fusion_k: int | None = None  # None = pure CE order; int = RRF(first-stage, CE) with this k
    rerank_weight: float = 1.0  # RRF weight of the CE list when rerank_fusion_k is set

    def fuse(self, bm25_ids: Sequence[str], dense_ids: Sequence[str]) -> Ranked:
        return rrf(
            [dense_ids, bm25_ids], k=self.rrf_k, depth=self.candidates, weights=[self.dense_weight, 1.0]
        )


@dataclass
class Hit:
    doc_id: str
    score: float
    text: str
    title: str = ""


@dataclass
class SearchResult:
    hits: list[Hit]
    timings_ms: dict[str, float] = field(default_factory=dict)


class HybridRetriever:
    def __init__(
        self,
        bm25: BM25Index,
        dense: QdrantDenseIndex,
        embedder: Embedder,
        reranker: Reranker | None = None,
        docs: DocStore | None = None,
        config: RetrieverConfig | None = None,
    ) -> None:
        self.bm25, self.dense, self.embedder, self.reranker = bm25, dense, embedder, reranker
        self.docs: DocStore = docs if docs is not None else dense
        self.config = config or RetrieverConfig()

    def ranked(self, query: str, mode: Mode, timings: dict[str, float] | None = None) -> Ranked:
        t = timings if timings is not None else {}
        cfg = self.config
        bm25_hits: Ranked = []
        dense_hits: Ranked = []
        if mode in ("bm25", "hybrid", "hybrid_rerank"):
            t0 = time.perf_counter()
            bm25_hits = self.bm25.search(query, k=cfg.candidates)
            t["bm25"] = (time.perf_counter() - t0) * 1e3
        if mode in ("dense", "hybrid", "hybrid_rerank"):
            t0 = time.perf_counter()
            qv = self.embedder.encode_queries([query])[0]
            t["embed_query"] = (time.perf_counter() - t0) * 1e3
            t0 = time.perf_counter()
            dense_hits = self.dense.search(qv, k=cfg.candidates)
            t["dense_search"] = (time.perf_counter() - t0) * 1e3
        if mode == "bm25":
            return bm25_hits
        if mode == "dense":
            return dense_hits
        t0 = time.perf_counter()
        fused = cfg.fuse(ranked_ids(bm25_hits), ranked_ids(dense_hits))
        t["fusion"] = (time.perf_counter() - t0) * 1e3
        if mode == "hybrid":
            return fused
        if self.reranker is None:
            raise ValueError("mode 'hybrid_rerank' needs a reranker")
        t0 = time.perf_counter()
        head = [d for d, _ in fused[: cfg.rerank_depth]]
        texts = self.docs.fetch(head)
        t["fetch_text"] = (time.perf_counter() - t0) * 1e3
        t0 = time.perf_counter()
        scores = self.reranker.score_pairs([(query, texts[d].text if d in texts else "") for d in head])
        t["rerank"] = (time.perf_counter() - t0) * 1e3
        return apply_rerank(fused, scores, cfg.rerank_depth, cfg.rerank_fusion_k, cfg.rerank_weight)

    def search(self, query: str, mode: Mode = "hybrid_rerank", k: int = 10) -> SearchResult:
        timings: dict[str, float] = {}
        t_start = time.perf_counter()
        top = self.ranked(query, mode, timings)[:k]
        docs = self.docs.fetch([d for d, _ in top])
        hits = [
            Hit(d, s, docs[d].text if d in docs else "", docs[d].title if d in docs else "") for d, s in top
        ]
        timings["total"] = (time.perf_counter() - t_start) * 1e3
        return SearchResult(hits=hits, timings_ms=timings)
