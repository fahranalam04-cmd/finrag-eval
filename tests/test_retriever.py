"""Retrieval on the fixture corpus, and parity between the online and offline (batch) code paths."""

from __future__ import annotations

import pytest

from finrag.experiments.retrieval import fused_runs, reranked_runs
from finrag.retriever import MODES
from finrag.types import ranked_ids


@pytest.mark.parametrize("mode", MODES)
def test_every_mode_finds_the_obvious_passage(retriever, mode):
    res = retriever.search("Roth IRA withdrawals taxed in retirement", mode=mode, k=3)
    assert res.hits and "102" in [h.doc_id for h in res.hits]
    assert res.hits[0].text  # text is joined from the doc store
    assert "total" in res.timings_ms


def test_bm25_handles_stopword_only_and_oov_queries(retriever):
    assert retriever.bm25.search("the of and") == []
    assert retriever.bm25.search("zzzqqq") == []


def test_dense_scores_are_cosine_and_sorted(retriever):
    v = retriever.embedder.encode_queries(["emergency fund months"])[0]
    hits = retriever.dense.search(v, k=5)
    scores = [s for _, s in hits]
    assert scores == sorted(scores, reverse=True) and all(-1.0001 <= s <= 1.0001 for s in scores)
    assert retriever.dense.count() == 12


def test_online_and_offline_paths_agree(retriever, passages):
    """The batch evaluator (cached runs + RRF + rerank) must reproduce the serving retriever exactly."""
    cfg = retriever.config
    queries = ["What affects my credit score?", "index fund fees", "bond prices and interest rates"]
    qids = [str(i) for i in range(len(queries))]
    bm25 = dict(zip(qids, retriever.bm25.search_batch(queries, k=cfg.candidates), strict=True))
    qv = retriever.embedder.encode_queries(queries)
    dense = dict(zip(qids, retriever.dense.search_batch(qv, k=cfg.candidates), strict=True))
    fused = fused_runs(bm25, dense, cfg)
    text = {p.doc_id: p.text for p in passages}
    ce = {
        q: dict(
            zip(
                ranked_ids(fused[q]),
                retriever.reranker.score_pairs(
                    [(queries[int(q)], text[d]) for d in ranked_ids(fused[q])]
                ).tolist(),
                strict=True,
            )
        )
        for q in qids
    }
    reranked = reranked_runs(fused, ce, cfg.rerank_depth)
    for q, text_q in zip(qids, queries, strict=True):
        assert ranked_ids(retriever.ranked(text_q, "hybrid")) == ranked_ids(fused[q])
        assert ranked_ids(retriever.ranked(text_q, "hybrid_rerank")) == ranked_ids(reranked[q])
