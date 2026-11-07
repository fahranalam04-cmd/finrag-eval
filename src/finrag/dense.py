"""Dense vector store on Qdrant.

Runs in qdrant-client's embedded (local) mode: no server, persisted to disk. Note that embedded mode
answers queries by *exact* cosine search over all vectors (it does not build an HNSW graph); the HNSW
parameters below are what a Qdrant server would use when the same code is pointed at one via `url`.
At 57.6K x 384-d vectors exact search is cheap, and it means dense recall here has no ANN error.
"""

from __future__ import annotations

import uuid
from collections.abc import Sequence
from pathlib import Path

import numpy as np
from qdrant_client import QdrantClient, models

from finrag.data import Passage
from finrag.types import Ranked

_NS = uuid.UUID("6f1c2a52-5f0e-4a8b-9a51-2f8a3c1d9e77")


def point_id(doc_id: str) -> str:
    """Stable UUID per corpus id (Qdrant point ids must be ints or UUIDs)."""
    return str(uuid.uuid5(_NS, doc_id))


class QdrantDenseIndex:
    def __init__(self, client: QdrantClient, collection: str, server: bool = False) -> None:
        self.client, self.collection = client, collection
        # HNSW search-time ef only means something on a server; embedded mode warns and ignores it
        self._params = (lambda k: models.SearchParams(hnsw_ef=max(128, k))) if server else (lambda k: None)

    @classmethod
    def local(cls, path: Path, collection: str) -> QdrantDenseIndex:
        path.mkdir(parents=True, exist_ok=True)
        return cls(QdrantClient(path=str(path)), collection)

    @classmethod
    def in_memory(cls, collection: str = "test") -> QdrantDenseIndex:
        return cls(QdrantClient(location=":memory:"), collection)

    @classmethod
    def remote(cls, url: str, collection: str) -> QdrantDenseIndex:
        return cls(QdrantClient(url=url), collection, server=True)

    def build(self, passages: Sequence[Passage], vectors: np.ndarray, batch_size: int = 512) -> None:
        if len(passages) != len(vectors):
            raise ValueError("passages and vectors must align")
        if self.client.collection_exists(self.collection):
            self.client.delete_collection(self.collection)
        self.client.create_collection(
            self.collection,
            vectors_config=models.VectorParams(size=int(vectors.shape[1]), distance=models.Distance.COSINE),
            hnsw_config=models.HnswConfigDiff(m=16, ef_construct=128),
        )
        for start in range(0, len(passages), batch_size):
            chunk = passages[start : start + batch_size]
            self.client.upsert(
                self.collection,
                points=[
                    models.PointStruct(
                        id=point_id(p.doc_id),
                        vector=vectors[start + j].tolist(),
                        payload={"doc_id": p.doc_id, "title": p.title, "text": p.text},
                    )
                    for j, p in enumerate(chunk)
                ],
                wait=True,
            )

    def count(self) -> int:
        return int(self.client.count(self.collection, exact=True).count)

    def search_batch(self, vectors: np.ndarray, k: int = 100) -> list[Ranked]:
        reqs = [
            models.QueryRequest(query=v.tolist(), limit=k, with_payload=["doc_id"], params=self._params(k))
            for v in vectors
        ]
        out: list[Ranked] = []
        for start in range(0, len(reqs), 64):
            res = self.client.query_batch_points(self.collection, requests=reqs[start : start + 64])
            out.extend([(p.payload["doc_id"], float(p.score)) for p in r.points] for r in res)
        return out

    def search(self, vector: np.ndarray, k: int = 100) -> Ranked:
        res = self.client.query_points(
            self.collection,
            query=vector.tolist(),
            limit=k,
            with_payload=["doc_id"],
            search_params=self._params(k),
        )
        return [(p.payload["doc_id"], float(p.score)) for p in res.points]

    def fetch(self, doc_ids: Sequence[str]) -> dict[str, Passage]:
        """Look up passage text by corpus id (payload store doubles as the document store)."""
        if not doc_ids:
            return {}
        recs = self.client.retrieve(self.collection, ids=[point_id(d) for d in doc_ids], with_payload=True)
        out = {}
        for r in recs:
            p = r.payload or {}
            out[p["doc_id"]] = Passage(p["doc_id"], p.get("text", ""), p.get("title", ""))
        return out

    def close(self) -> None:
        self.client.close()
