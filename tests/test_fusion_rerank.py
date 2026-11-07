"""RRF and rerank-application logic."""

from __future__ import annotations

import pytest

from finrag.fusion import rrf
from finrag.rerank import TermOverlapReranker, apply_rerank


def test_rrf_hand_computed():
    fused = dict(rrf([["a", "b", "c"], ["b", "d"]], k=60))
    assert fused["b"] == pytest.approx(1 / 62 + 1 / 61)
    assert fused["a"] == pytest.approx(1 / 61)
    assert fused["d"] == pytest.approx(1 / 62)
    assert fused["c"] == pytest.approx(1 / 63)
    order = [d for d, _ in rrf([["a", "b", "c"], ["b", "d"]], k=60)]
    assert order == ["b", "a", "d", "c"]


def test_rrf_ties_prefer_first_list_and_weights_shift_order():
    # a is rank 1 in list 0, x is rank 1 in list 1 -> equal scores, list 0 wins the tie
    assert [d for d, _ in rrf([["a"], ["x"]])] == ["a", "x"]
    # up-weighting list 1 flips it
    assert [d for d, _ in rrf([["a"], ["x"]], weights=[1.0, 2.0])] == ["x", "a"]


def test_rrf_depth_and_duplicates():
    fused = dict(rrf([["a", "b", "c"], ["c"]], k=10, depth=2))
    assert "c" in fused and fused["c"] == pytest.approx(1 / 11)  # only via list 1 (cut from list 0)
    dup = dict(rrf([["a", "a", "b"]], k=10))
    assert dup["a"] == pytest.approx(1 / 11) and dup["b"] == pytest.approx(1 / 13)
    with pytest.raises(ValueError):
        rrf([["a"]], weights=[1.0, 2.0])


def test_apply_rerank_only_moves_head():
    cands = [("a", 0.9), ("b", 0.8), ("c", 0.7), ("d", 0.6)]
    out = apply_rerank(cands, [0.1, 0.5, 0.3], depth=3)
    assert [d for d, _ in out] == ["b", "c", "a", "d"]
    assert out[-1] == ("d", 0.6)
    with pytest.raises(ValueError):
        apply_rerank(cands, [0.1], depth=3)


def test_term_overlap_reranker():
    s = TermOverlapReranker().score_pairs([("index fund fees", "index fund"), ("index fund fees", "bonds")])
    assert s[0] > s[1]


def test_apply_rerank_rank_fusion_keeps_first_stage_evidence():
    cands = [("a", 0.9), ("b", 0.8), ("c", 0.7), ("d", 0.6)]
    scores = [0.1, 0.9, 0.5]  # CE order: b, c, a
    # k=1, equal weights: a = 1/2 + 1/4, b = 1/3 + 1/2, c = 1/4 + 1/3 -> b, a, c
    out = apply_rerank(cands, scores, depth=3, fusion_k=1, weight=1.0)
    assert [d for d, _ in out] == ["b", "a", "c", "d"]
    assert out[1][1] == pytest.approx(1 / 2 + 1 / 4)
    # a tiny CE weight leaves the first-stage order intact
    assert [d for d, _ in apply_rerank(cands, scores, depth=3, fusion_k=1, weight=0.01)] == [
        "a",
        "b",
        "c",
        "d",
    ]
