"""Second-stage reranking of the fused candidate list with a cross-encoder."""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Protocol

import numpy as np

from finrag.types import Ranked


class Reranker(Protocol):
    name: str

    def score_pairs(self, pairs: Sequence[tuple[str, str]]) -> np.ndarray: ...


class TermOverlapReranker:
    """Model-free stand-in for tests: score = share of query terms present in the passage."""

    name = "term-overlap"
    _tok = re.compile(r"[a-z0-9]+")

    def score_pairs(self, pairs: Sequence[tuple[str, str]]) -> np.ndarray:
        out = []
        for q, p in pairs:
            qt, pt = set(self._tok.findall(q.lower())), set(self._tok.findall(p.lower()))
            out.append(len(qt & pt) / max(len(qt), 1))
        return np.asarray(out, dtype=np.float32)


class CrossEncoderReranker:
    def __init__(
        self,
        hf_id: str,
        name: str,
        device: str | None = None,
        max_length: int = 512,
        batch_size: int = 32,
        fp16: bool = True,
    ) -> None:
        import torch
        from sentence_transformers import CrossEncoder

        from finrag.config import revision

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = CrossEncoder(hf_id, device=self.device, max_length=max_length, revision=revision(hf_id))
        if fp16 and self.device == "cuda":
            self.model.model.half()
        self.name, self.hf_id, self.batch_size = name, hf_id, batch_size

    def score_pairs(self, pairs: Sequence[tuple[str, str]]) -> np.ndarray:
        if not pairs:
            return np.zeros(0, dtype=np.float32)
        s = self.model.predict(
            [list(p) for p in pairs],
            batch_size=self.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return np.asarray(s, dtype=np.float32).reshape(len(pairs), -1)[:, -1]


def apply_rerank(candidates: Ranked, scores: Sequence[float], depth: int) -> Ranked:
    """Re-order the first `depth` candidates by cross-encoder score; the tail keeps its fused order
    (so Recall@depth is unchanged by reranking and only the head ordering moves)."""
    depth = min(depth, len(candidates))
    if len(scores) < depth:
        raise ValueError(f"need {depth} scores, got {len(scores)}")
    head = sorted(
        ((candidates[i][0], float(scores[i]), i) for i in range(depth)), key=lambda x: (-x[1], x[2])
    )
    return [(d, s) for d, s, _ in head] + list(candidates[depth:])
