"""IR metrics against hand-computed values."""

from __future__ import annotations

import math

import pytest

from finrag.metrics import (
    bootstrap_ci,
    evaluate_run,
    mrr_at_k,
    ndcg_at_k,
    paired_bootstrap,
    precision_at_k,
    recall_at_k,
)

RELS = {"a": 1, "c": 1}  # two relevant docs
RANKED = ["b", "a", "d", "c", "e"]  # relevant at ranks 2 and 4


def test_ndcg_hand_computed():
    # DCG = 1/log2(3) + 1/log2(5); IDCG = 1/log2(2) + 1/log2(3)
    dcg = 1 / math.log2(3) + 1 / math.log2(5)
    idcg = 1 + 1 / math.log2(3)
    assert ndcg_at_k(RANKED, RELS, 10) == pytest.approx(dcg / idcg)
    assert ndcg_at_k(RANKED, RELS, 10) == pytest.approx(0.6509209, abs=1e-6)
    # cut at 3: only rank-2 hit counts, ideal still has two relevant docs within top 3
    assert ndcg_at_k(RANKED, RELS, 3) == pytest.approx((1 / math.log2(3)) / idcg)


def test_ndcg_graded_relevance():
    rels = {"x": 2, "y": 1}
    # ranked y, x: DCG = 1/1 + 2/log2(3); IDCG = 2/1 + 1/log2(3)
    expected = (1 + 2 / math.log2(3)) / (2 + 1 / math.log2(3))
    assert ndcg_at_k(["y", "x"], rels, 10) == pytest.approx(expected)
    assert ndcg_at_k(["x", "y"], rels, 10) == pytest.approx(1.0)


def test_recall_mrr_precision():
    assert recall_at_k(RANKED, RELS, 2) == 0.5
    assert recall_at_k(RANKED, RELS, 4) == 1.0
    assert recall_at_k(RANKED, RELS, 1) == 0.0
    assert mrr_at_k(RANKED, RELS, 10) == 0.5
    assert mrr_at_k(RANKED, RELS, 1) == 0.0
    assert precision_at_k(RANKED, RELS, 4) == 0.5


def test_duplicates_do_not_inflate_scores():
    assert recall_at_k(["a", "a", "a"], RELS, 3) == 0.5
    assert ndcg_at_k(["a", "a"], {"a": 1}, 10) == pytest.approx(1.0)


def test_no_hits_and_empty_run():
    assert ndcg_at_k(["z"], RELS, 10) == 0.0
    assert mrr_at_k([], RELS, 10) == 0.0


def test_evaluate_run_counts_missing_queries_as_zero_and_skips_unjudged():
    qrels = {"q1": {"a": 1}, "q2": {"b": 1}, "q3": {"c": 0}}  # q3 has no positive judgement
    run = {"q1": ["a"], "q4": ["x"]}  # q2 missing from run, q4 not in qrels
    ev = evaluate_run(run, qrels, metrics=("ndcg@10", "recall@10"))
    assert ev.n_queries == 2
    assert ev.mean["ndcg@10"] == pytest.approx(0.5)
    assert ev.per_query["recall@10"] == {"q1": 1.0, "q2": 0.0}


def test_bootstrap_ci_brackets_mean():
    xs = [0.0, 1.0] * 50
    lo, hi = bootstrap_ci(xs, n_boot=2000, seed=1)
    assert lo < 0.5 < hi


def test_paired_bootstrap_detects_consistent_lift_and_null():
    a = [0.1 * (i % 5) for i in range(200)]
    b = [x + 0.2 for x in a]
    r = paired_bootstrap(a, b, n_boot=2000)
    assert r["diff"] == pytest.approx(0.2)
    assert r["ci95_low"] == pytest.approx(0.2) and r["p_value"] < 0.01
    null = paired_bootstrap(a, a, n_boot=2000)
    assert null["diff"] == 0.0 and null["p_value"] == 1.0
    with pytest.raises(ValueError):
        paired_bootstrap([1.0], [1.0, 2.0])
