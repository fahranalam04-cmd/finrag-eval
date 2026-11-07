"""Paths, pinned sources and model ids. Heavy artefacts live outside the repo (env-configurable)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
REPORTS_DIR = REPO_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# Pinned Hugging Face dataset revisions (commit shas) so the numbers are reproducible.
FIQA_REPO = "BeIR/fiqa"
FIQA_REVISION = "979c07a7cb5ccc6ca009792241fa1250b98055dd"
FIQA_QRELS_REPO = "BeIR/fiqa-qrels"
FIQA_QRELS_REVISION = "252958f2d646e22cab6d0c72dd3f0d5de6d0655a"

# Embedders compared on dev. Prefix strings follow each model card.
EMBEDDERS: dict[str, dict[str, str]] = {
    "bge-small": {
        "hf_id": "BAAI/bge-small-en-v1.5",
        "query_prefix": "Represent this sentence for searching relevant passages: ",
        "passage_prefix": "",
    },
    "bge-base": {
        "hf_id": "BAAI/bge-base-en-v1.5",
        "query_prefix": "Represent this sentence for searching relevant passages: ",
        "passage_prefix": "",
    },
}

RERANKERS: dict[str, str] = {
    "minilm-l6": "cross-encoder/ms-marco-MiniLM-L6-v2",
    "bge-reranker-base": "BAAI/bge-reranker-base",
}

GENERATOR_HF_ID = "Qwen/Qwen2.5-1.5B-Instruct"
NLI_HF_ID = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"

SPLITS = ("train", "dev", "test")

# Pinned model revisions (HF commit shas, fetched 2026-11-07) + licenses, for reproducibility.
MODEL_REVISIONS: dict[str, str] = {
    "BAAI/bge-small-en-v1.5": "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a",
    "BAAI/bge-base-en-v1.5": "a5beb1e3e68b9ab74eb54cfd186867f64f240e1a",
    "cross-encoder/ms-marco-MiniLM-L6-v2": "233902d25c440f23af6f7d6e94d2946bac0bee0a",
    "BAAI/bge-reranker-base": "2cfc18c9415c912f9d8155881c133215df768a70",
    "Qwen/Qwen2.5-1.5B-Instruct": "989aa7980e4cf806f80c7fef2b1adb7bc71aa306",
    "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli": "6f5cf0a2b59cabb106aca4c287eed12e357e90eb",
}
MODEL_LICENSES: dict[str, str] = {
    "BAAI/bge-small-en-v1.5": "MIT",
    "BAAI/bge-base-en-v1.5": "MIT",
    "cross-encoder/ms-marco-MiniLM-L6-v2": "Apache-2.0",
    "BAAI/bge-reranker-base": "MIT",
    "Qwen/Qwen2.5-1.5B-Instruct": "Apache-2.0",
    "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli": "MIT",
}


def revision(hf_id: str) -> str | None:
    """Pinned revision for a known model id (None = latest, e.g. for ad-hoc models)."""
    return MODEL_REVISIONS.get(hf_id)


def default_data_dir() -> Path:
    if v := os.environ.get("FINRAG_DATA_DIR"):
        return Path(v)
    return Path(os.environ.get("ML_DATA_ROOT", "D:/ml-portfolio/data")) / "finrag"


def default_models_dir() -> Path:
    if v := os.environ.get("FINRAG_MODELS_DIR"):
        return Path(v)
    return Path(os.environ.get("ML_MODELS_ROOT", "D:/ml-portfolio/models")) / "finrag"


@dataclass(frozen=True)
class Paths:
    """Where data / indexes / caches live. Override with FINRAG_DATA_DIR / FINRAG_MODELS_DIR."""

    data_dir: Path = field(default_factory=default_data_dir)
    models_dir: Path = field(default_factory=default_models_dir)

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def corpus_jsonl(self) -> Path:
        return self.data_dir / "corpus.jsonl"

    @property
    def queries_jsonl(self) -> Path:
        return self.data_dir / "queries.jsonl"

    def qrels_tsv(self, split: str) -> Path:
        return self.data_dir / "qrels" / f"{split}.tsv"

    @property
    def bm25_dir(self) -> Path:
        return self.models_dir / "bm25"

    def embeddings_npy(self, embedder: str) -> Path:
        return self.models_dir / "embeddings" / f"{embedder}.corpus.npy"

    def query_embeddings_npy(self, embedder: str, split: str) -> Path:
        return self.models_dir / "embeddings" / f"{embedder}.queries.{split}.npy"

    @property
    def qdrant_dir(self) -> Path:
        return self.models_dir / "qdrant"

    @property
    def runs_dir(self) -> Path:
        return self.models_dir / "runs"
