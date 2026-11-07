"""Sparse lexical retrieval: BM25 (Lucene variant) via bm25s with an English Snowball stemmer."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import bm25s
import Stemmer

from finrag.types import Ranked


class BM25Index:
    def __init__(self, k1: float = 1.2, b: float = 0.75) -> None:
        self.k1, self.b = k1, b
        self._stemmer = Stemmer.Stemmer("english")
        self._bm25: bm25s.BM25 | None = None
        self.doc_ids: list[str] = []

    def tokenize(self, texts: Sequence[str]) -> list[list[str]]:
        """Lower-case, English stop-word removal, Snowball stemming (same pipeline for docs and queries)."""
        return bm25s.tokenize(
            list(texts), stopwords="en", stemmer=self._stemmer, return_ids=False, show_progress=False
        )

    def build(
        self, doc_ids: Sequence[str], texts: Sequence[str], tokens: list[list[str]] | None = None
    ) -> BM25Index:
        """Index passages; pass pre-computed `tokens` to reuse tokenisation across a parameter grid."""
        if len(doc_ids) != len(texts):
            raise ValueError("doc_ids and texts must align")
        self.doc_ids = list(doc_ids)
        self._bm25 = bm25s.BM25(k1=self.k1, b=self.b, method="lucene")
        self._bm25.index(tokens if tokens is not None else self.tokenize(texts), show_progress=False)
        return self

    def save(self, directory: Path) -> None:
        assert self._bm25 is not None
        directory.mkdir(parents=True, exist_ok=True)
        self._bm25.save(str(directory), show_progress=False)
        (directory / "doc_ids.json").write_text(json.dumps(self.doc_ids), encoding="utf-8")

    @classmethod
    def load(cls, directory: Path) -> BM25Index:
        idx = cls()
        idx._bm25 = bm25s.BM25.load(str(directory))
        idx.k1, idx.b = idx._bm25.k1, idx._bm25.b
        idx.doc_ids = json.loads((directory / "doc_ids.json").read_text(encoding="utf-8"))
        return idx

    def search_batch(self, queries: Sequence[str], k: int = 100, n_threads: int = 4) -> list[Ranked]:
        assert self._bm25 is not None, "index not built/loaded"
        k = min(k, len(self.doc_ids))
        tokens = self.tokenize(queries)
        out: list[Ranked] = [[] for _ in queries]
        # bm25s cannot score an empty token list (all stopwords / OOV): those queries get no hits.
        vocab = self._bm25.vocab_dict
        live = [i for i, t in enumerate(tokens) if any(tok in vocab for tok in t)]
        if not live:
            return out
        docs, scores = self._bm25.retrieve(
            [[tok for tok in tokens[i] if tok in vocab] for i in live],
            k=k,
            show_progress=False,
            n_threads=n_threads,
        )
        for row, i in enumerate(live):
            out[i] = [
                (self.doc_ids[int(d)], float(s)) for d, s in zip(docs[row], scores[row], strict=True) if s > 0
            ]
        return out

    def search(self, query: str, k: int = 100) -> Ranked:
        return self.search_batch([query], k=k, n_threads=1)[0]
