"""Reciprocal Rank Fusion (Cormack, Clarke & Buettcher, SIGIR 2009)."""

from __future__ import annotations

from collections.abc import Sequence

from finrag.types import Ranked


def rrf(
    rankings: Sequence[Sequence[str]],
    k: int = 60,
    depth: int | None = None,
    weights: Sequence[float] | None = None,
) -> Ranked:
    """Fuse ranked id lists: score(d) = sum_i w_i / (k + rank_i(d)), ranks 1-based.

    Only the first `depth` ids of each list participate. Ties are broken by the best individual
    (rank, list position) so the order is deterministic and favours the earlier-listed ranking.
    """
    if weights is None:
        weights = [1.0] * len(rankings)
    if len(weights) != len(rankings):
        raise ValueError("one weight per ranking")
    scores: dict[str, float] = {}
    best: dict[str, tuple[int, int]] = {}
    for li, (w, ranking) in enumerate(zip(weights, rankings, strict=True)):
        seen: set[str] = set()
        for rank, d in enumerate(ranking[:depth] if depth else ranking, start=1):
            if d in seen:  # ignore duplicates inside one list
                continue
            seen.add(d)
            scores[d] = scores.get(d, 0.0) + w / (k + rank)
            best[d] = min(best.get(d, (rank, li)), (rank, li))
    return sorted(scores.items(), key=lambda x: (-x[1], best[x[0]]))
