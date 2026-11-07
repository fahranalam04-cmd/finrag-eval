"""Shared lightweight types."""

from __future__ import annotations

Ranked = list[tuple[str, float]]  # (doc_id, score), best first


def ranked_ids(ranked: Ranked) -> list[str]:
    return [d for d, _ in ranked]
