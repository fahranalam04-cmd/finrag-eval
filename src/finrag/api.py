"""FastAPI service: POST /search and POST /answer over the tuned hybrid retriever.

Run against the real indexes:   uv run uvicorn finrag.api:create_default_app --factory --port 8000
(env: FINRAG_DEVICE=cuda|cpu, FINRAG_GENERATOR=0 to serve /search only)
"""

from __future__ import annotations

import os

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from finrag import __version__
from finrag.config import Paths
from finrag.rag import RAGPipeline
from finrag.retriever import MODES, Hit, Mode


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    k: int = Field(10, ge=1, le=100)
    mode: Mode | None = None  # None = the dev-selected pipeline


class PassageOut(BaseModel):
    rank: int
    doc_id: str
    score: float
    text: str


class SearchResponse(BaseModel):
    query: str
    mode: Mode
    hits: list[PassageOut]
    timings_ms: dict[str, float]


class AnswerRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    k_context: int = Field(5, ge=1, le=10)
    mode: Mode | None = None  # None = the dev-selected pipeline


class CitationOut(BaseModel):
    doc_id: str
    in_context: bool


class AnswerResponse(BaseModel):
    query: str
    mode: Mode
    answer: str
    abstained: bool
    citations: list[CitationOut]
    passages: list[PassageOut]
    timings_ms: dict[str, float]


class HealthResponse(BaseModel):
    status: str
    default_mode: str
    version: str
    modes: list[str]
    reranker: str | None
    generator: str | None


def _passages(hits: list[Hit]) -> list[PassageOut]:
    return [PassageOut(rank=i + 1, doc_id=h.doc_id, score=h.score, text=h.text) for i, h in enumerate(hits)]


def create_app(pipeline: RAGPipeline) -> FastAPI:
    app = FastAPI(
        title="finrag-eval", version=__version__, description="Hybrid RAG over FiQA-2018 with citations"
    )

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        rr = pipeline.retriever.reranker
        modes = [m for m in MODES if m != "hybrid_rerank" or rr is not None]
        return HealthResponse(
            status="ok",
            default_mode=pipeline.default_mode,
            version=__version__,
            modes=modes,
            reranker=getattr(rr, "name", None),
            generator=getattr(pipeline.generator, "name", None),
        )

    @app.post("/search", response_model=SearchResponse)
    def search(req: SearchRequest) -> SearchResponse:
        try:
            mode = req.mode or pipeline.default_mode
            res = pipeline.search(req.query, mode, req.k)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e
        return SearchResponse(query=req.query, mode=mode, hits=_passages(res.hits), timings_ms=res.timings_ms)

    @app.post("/answer", response_model=AnswerResponse)
    def answer(req: AnswerRequest) -> AnswerResponse:
        if pipeline.generator is None:
            raise HTTPException(status_code=503, detail="generator not loaded (FINRAG_GENERATOR=0)")
        try:
            mode = req.mode or pipeline.default_mode
            a = pipeline.answer(req.query, mode, req.k_context)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e
        return AnswerResponse(
            query=a.query,
            mode=mode,
            answer=a.answer,
            abstained=a.abstained,
            citations=[CitationOut(doc_id=c.doc_id, in_context=c.in_context) for c in a.citations],
            passages=_passages(a.passages),
            timings_ms=a.timings_ms,
        )

    return app


def create_default_app() -> FastAPI:
    """Factory for uvicorn: loads the dev-tuned indexes/models from disk."""
    from finrag.rag import load_pipeline

    device = os.environ.get("FINRAG_DEVICE", "cuda")
    with_gen = os.environ.get("FINRAG_GENERATOR", "1") != "0"
    return create_app(load_pipeline(Paths(), device=device, with_generator=with_gen))
