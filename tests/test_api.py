"""API contract with a fixture pipeline (fake embedder / reranker / generator, no downloads)."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from finrag.api import create_app
from finrag.rag import RAGPipeline


@pytest.fixture(scope="module")
def client(pipeline) -> TestClient:
    return TestClient(create_app(pipeline))


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok" and "hybrid_rerank" in body["modes"]
    assert body["generator"] == "fake-extractive" and body["reranker"] == "term-overlap"


def test_search_contract(client):
    r = client.post("/search", json={"query": "credit score factors", "k": 3, "mode": "hybrid"})
    assert r.status_code == 200
    body = r.json()
    assert body["mode"] == "hybrid" and len(body["hits"]) == 3
    assert [h["rank"] for h in body["hits"]] == [1, 2, 3]
    assert {"doc_id", "score", "text"} <= set(body["hits"][0])
    assert body["hits"][0]["doc_id"] in {"105", "106"}
    assert "fusion" in body["timings_ms"]


def test_answer_contract_citations_point_at_returned_passages(client):
    r = client.post("/answer", json={"query": "How big should an emergency fund be?", "k_context": 3})
    assert r.status_code == 200
    body = r.json()
    assert not body["abstained"] and body["citations"]
    passage_ids = {p["doc_id"] for p in body["passages"]}
    assert all(c["in_context"] and c["doc_id"] in passage_ids for c in body["citations"])
    assert "generate" in body["timings_ms"] and len(body["passages"]) == 3


@pytest.mark.parametrize(
    "payload",
    [{"query": ""}, {"query": "x", "k": 0}, {"query": "x", "k": 101}, {"query": "x", "mode": "magic"}],
)
def test_search_validation(client, payload):
    assert client.post("/search", json=payload).status_code == 422


def test_answer_validation(client):
    assert client.post("/answer", json={"query": "x", "k_context": 11}).status_code == 422


def test_answer_without_generator_is_503(retriever):
    c = TestClient(create_app(RAGPipeline(retriever, generator=None)))
    assert c.post("/answer", json={"query": "fees"}).status_code == 503
    assert c.post("/search", json={"query": "fees"}).status_code == 200
