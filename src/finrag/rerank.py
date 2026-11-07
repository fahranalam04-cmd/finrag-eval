"""Second-stage reranking of the fused candidate list with a cross-encoder."""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Protocol

import numpy as np

from finrag.fusion import rrf
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


def apply_rerank(
    candidates: Ranked,
    scores: Sequence[float],
    depth: int,
    fusion_k: int | None = None,
    weight: float = 1.0,
) -> Ranked:
    """Re-order the first `depth` candidates with cross-encoder scores; the tail keeps its fused order
    (so Recall@depth is unchanged and only the head ordering moves).

    fusion_k=None : pure rerank, the head is sorted by cross-encoder score.
    fusion_k=k    : rank fusion, RRF(first-stage head order, cross-encoder order) with the CE list
                    weighted by `weight`, so first-stage evidence is kept rather than discarded.
    """
    depth = min(depth, len(candidates))
    if len(scores) < depth:
        raise ValueError(f"need {depth} scores, got {len(scores)}")
    by_ce = sorted(range(depth), key=lambda i: (-float(scores[i]), i))
    if fusion_k is None:
        head = [(candidates[i][0], float(scores[i])) for i in by_ce]
    else:
        ids = [candidates[i][0] for i in range(depth)]
        head = rrf([ids, [ids[i] for i in by_ce]], k=fusion_k, weights=[1.0, weight])
    return head + list(candidates[depth:])
