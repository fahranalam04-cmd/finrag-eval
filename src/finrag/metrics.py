"""IR metrics implemented from their definitions (cross-checked against pytrec_eval in reports/).

Conventions match trec_eval / BEIR: linear gain (rel), log2(rank + 1) discount, ideal DCG from the
judged relevance grades, per-query scores averaged over every query in the qrels (a query with no
retrieved list scores 0). Recall@k = |relevant in top k| / |relevant|.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

Run = Mapping[str, Sequence[str]]  # query_id -> ranked doc ids (best first)
QrelsT = Mapping[str, Mapping[str, int]]

DEFAULT_METRICS = ("ndcg@10", "recall@10", "recall@100", "mrr@10")


def _dedupe(ranked: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    out = []
    for d in ranked:
        if d not in seen:
            seen.add(d)
            out.append(d)
    return out


def dcg(gains: Sequence[float]) -> float:
    return sum(g / math.log2(i + 2) for i, g in enumerate(gains))


def ndcg_at_k(ranked: Sequence[str], rels: Mapping[str, int], k: int) -> float:
    ranked = _dedupe(ranked)
    ideal = sorted((r for r in rels.values() if r > 0), reverse=True)[:k]
    idcg = dcg(ideal)
    if idcg == 0:
        return 0.0
    return dcg([max(rels.get(d, 0), 0) for d in ranked[:k]]) / idcg


def recall_at_k(ranked: Sequence[str], rels: Mapping[str, int], k: int) -> float:
    relevant = {d for d, r in rels.items() if r > 0}
    if not relevant:
        return 0.0
    return len(relevant.intersection(_dedupe(ranked)[:k])) / len(relevant)


def mrr_at_k(ranked: Sequence[str], rels: Mapping[str, int], k: int) -> float:
    for i, d in enumerate(_dedupe(ranked)[:k]):
        if rels.get(d, 0) > 0:
            return 1.0 / (i + 1)
    return 0.0


def precision_at_k(ranked: Sequence[str], rels: Mapping[str, int], k: int) -> float:
    top = _dedupe(ranked)[:k]
    return sum(rels.get(d, 0) > 0 for d in top) / k if k > 0 else 0.0


_FUNCS = {"ndcg": ndcg_at_k, "recall": recall_at_k, "mrr": mrr_at_k, "precision": precision_at_k}


def score_query(ranked: Sequence[str], rels: Mapping[str, int], metric: str) -> float:
    name, k = metric.split("@")
    return _FUNCS[name](ranked, rels, int(k))


@dataclass
class EvalResult:
    mean: dict[str, float]
    per_query: dict[str, dict[str, float]]  # metric -> {qid: score}
    n_queries: int


def evaluate_run(run: Run, qrels: QrelsT, metrics: Sequence[str] = DEFAULT_METRICS) -> EvalResult:
    """Score a run against qrels. Only queries with >= 1 relevant judgement are evaluated."""
    qids = [q for q, rels in qrels.items() if any(r > 0 for r in rels.values())]
    per_query: dict[str, dict[str, float]] = {m: {} for m in metrics}
    for q in qids:
        ranked = run.get(q, [])
        for m in metrics:
            per_query[m][q] = score_query(ranked, qrels[q], m)
    mean = {m: float(np.mean(list(v.values()))) if v else 0.0 for m, v in per_query.items()}
    return EvalResult(mean=mean, per_query=per_query, n_queries=len(qids))


def bootstrap_ci(
    values: Sequence[float], n_boot: int = 10_000, seed: int = 0, alpha: float = 0.05
) -> tuple[float, float]:
    """Percentile bootstrap CI of the mean over queries."""
    x = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(n_boot, len(x)))
    means = x[idx].mean(axis=1)
    return float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def paired_bootstrap(
    a: Sequence[float], b: Sequence[float], n_boot: int = 10_000, seed: int = 0
) -> dict[str, float]:
    """Paired bootstrap over queries for mean(b) - mean(a).

    Returns the observed difference, a 95% percentile CI and a two-sided p-value: the share of the
    centred (null) bootstrap distribution at least as extreme as the observed difference, with +1
    smoothing so p is never exactly 0."""
    x, y = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if x.shape != y.shape:
        raise ValueError("paired samples must align")
    d = y - x
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(n_boot, len(d)))
    boot = d[idx].mean(axis=1)
    obs = float(d.mean())
    centred = boot - obs
    p = float(((np.abs(centred) >= abs(obs)).sum() + 1) / (n_boot + 1))
    return {
        "diff": obs,
        "ci95_low": float(np.quantile(boot, 0.025)),
        "ci95_high": float(np.quantile(boot, 0.975)),
        "p_value": p,
    }
