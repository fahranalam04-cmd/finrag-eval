"""Shared fixtures: a 12-passage corpus indexed with BM25 + in-memory Qdrant, model-free fakes."""

from __future__ import annotations

from pathlib import Path

import pytest

from finrag.bm25 import BM25Index
from finrag.data import Passage, clean_text, load_corpus, load_qrels, load_queries
from finrag.dense import QdrantDenseIndex
from finrag.embedders import HashingEmbedder
from finrag.generation import FakeGenerator
from finrag.rag import RAGPipeline
from finrag.rerank import TermOverlapReranker
from finrag.retriever import HybridRetriever, RetrieverConfig

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="session")
def passages() -> list[Passage]:
    return [Passage(p.doc_id, clean_text(p.text), p.title) for p in load_corpus(FIXTURES / "corpus.jsonl")]


@pytest.fixture(scope="session")
def fixture_queries():
    return load_queries(FIXTURES / "queries.jsonl")


@pytest.fixture(scope="session")
def fixture_qrels():
    return {s: load_qrels(FIXTURES / f"{s}.tsv") for s in ("dev", "test")}


@pytest.fixture(scope="session")
def retriever(passages) -> HybridRetriever:
    bm25 = BM25Index().build([p.doc_id for p in passages], [p.text for p in passages])
    emb = HashingEmbedder(dim=128)
    dense = QdrantDenseIndex.in_memory("fixture")
    dense.build(passages, emb.encode_passages([p.text for p in passages]))
    cfg = RetrieverConfig(candidates=10, rrf_k=60, dense_weight=1.0, rerank_depth=5)
    return HybridRetriever(bm25, dense, emb, TermOverlapReranker(), config=cfg)


@pytest.fixture(scope="session")
def pipeline(retriever) -> RAGPipeline:
    return RAGPipeline(retriever, FakeGenerator())
