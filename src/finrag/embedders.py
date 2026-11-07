"""Dense text encoders behind a tiny protocol, so tests can swap in a model-free fake."""

from __future__ import annotations

import re
import zlib
from collections.abc import Sequence
from typing import Protocol

import numpy as np


class Embedder(Protocol):
    name: str
    dim: int

    def encode_queries(self, texts: Sequence[str]) -> np.ndarray: ...

    def encode_passages(self, texts: Sequence[str]) -> np.ndarray: ...


def _l2(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return (x / np.maximum(n, 1e-12)).astype(np.float32)


class HashingEmbedder:
    """Deterministic bag-of-words feature hashing (crc32, signed). Test/CI stand-in, not a real model."""

    _tok = re.compile(r"[a-z0-9]+")

    def __init__(self, dim: int = 256, name: str = "hashing") -> None:
        self.dim, self.name = dim, name

    def _encode(self, texts: Sequence[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            for tok in self._tok.findall(t.lower()):
                h = zlib.crc32(tok.encode())
                out[i, h % self.dim] += 1.0 if (h >> 16) & 1 else -1.0
            if not out[i].any():  # keep empty texts off the zero vector (cosine undefined)
                out[i, 0] = 1.0
        return _l2(out)

    def encode_queries(self, texts: Sequence[str]) -> np.ndarray:
        return self._encode(texts)

    def encode_passages(self, texts: Sequence[str]) -> np.ndarray:
        return self._encode(texts)


class SentenceTransformerEmbedder:
    """Wraps a sentence-transformers bi-encoder (e.g. BAAI/bge-small-en-v1.5); cosine-normalised outputs."""

    def __init__(
        self,
        hf_id: str,
        name: str,
        query_prefix: str = "",
        passage_prefix: str = "",
        device: str | None = None,
        batch_size: int = 64,
        max_seq_length: int = 512,
        fp16: bool = True,
    ) -> None:
        import torch
        from sentence_transformers import SentenceTransformer

        from finrag.config import revision

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = SentenceTransformer(hf_id, device=self.device, revision=revision(hf_id))
        self.model.max_seq_length = max_seq_length
        if fp16 and self.device == "cuda":
            self.model.half()
        self.name, self.hf_id = name, hf_id
        self.query_prefix, self.passage_prefix = query_prefix, passage_prefix
        self.batch_size = batch_size
        self.dim = int(self.model.get_embedding_dimension())

    def _encode(self, texts: Sequence[str], show_progress: bool) -> np.ndarray:
        emb = self.model.encode(
            list(texts),
            batch_size=self.batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=show_progress,
        )
        return np.asarray(emb, dtype=np.float32)

    def encode_queries(self, texts: Sequence[str], show_progress: bool = False) -> np.ndarray:
        return self._encode([self.query_prefix + t for t in texts], show_progress)

    def encode_passages(self, texts: Sequence[str], show_progress: bool = False) -> np.ndarray:
        return self._encode([self.passage_prefix + t for t in texts], show_progress)
