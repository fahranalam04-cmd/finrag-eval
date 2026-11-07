"""Shared helpers for experiment stages: data access, run caching, report writing."""

from __future__ import annotations

import datetime as dt
import json
import os
import platform
from functools import lru_cache
from pathlib import Path
from typing import Any

from finrag.config import REPORTS_DIR, Paths
from finrag.data import Passage, Qrels, Query, load_corpus, load_qrels, load_queries, split_queries
from finrag.types import Ranked

TUNING_SPLIT = "dev"
REPORT_SPLIT = "test"


def set_cpu_threads(n: int = 4) -> None:
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS"):
        os.environ.setdefault(var, str(n))
    try:
        import torch

        torch.set_num_threads(n)
    except ImportError:
        pass


@lru_cache(maxsize=1)
def corpus(paths: Paths | None = None) -> list[Passage]:
    return load_corpus((paths or Paths()).corpus_jsonl)


@lru_cache(maxsize=1)
def corpus_by_id(paths: Paths | None = None) -> dict[str, Passage]:
    return {p.doc_id: p for p in corpus(paths)}


@lru_cache(maxsize=1)
def all_queries(paths: Paths | None = None) -> dict[str, Query]:
    return load_queries((paths or Paths()).queries_jsonl)


@lru_cache(maxsize=4)
def qrels(split: str, paths: Paths | None = None) -> Qrels:
    return load_qrels((paths or Paths()).qrels_tsv(split))


def queries(split: str, paths: Paths | None = None) -> list[Query]:
    return split_queries(all_queries(paths), qrels(split, paths))


def require_tuning_split(split: str) -> None:
    """Leakage guard: model / hyper-parameter selection may only ever look at the dev split."""
    if split != TUNING_SPLIT:
        raise PermissionError(f"tuning is only allowed on '{TUNING_SPLIT}', got '{split}'")


# ------------------------------------------------------------------------------------------ run cache


def save_runs(path: Path, runs: dict[str, Ranked]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({q: [[d, round(s, 6)] for d, s in r] for q, r in runs.items()}), encoding="utf-8"
    )


def load_runs(path: Path) -> dict[str, Ranked]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return {q: [(d, float(s)) for d, s in r] for q, r in raw.items()}


def run_path(paths: Paths, kind: str, split: str) -> Path:
    return paths.runs_dir / f"{kind}.{split}.json"


# ------------------------------------------------------------------------------------------- reports


def hardware(gpu: bool = False) -> dict[str, Any]:
    info: dict[str, Any] = {
        "cpu": "Intel i7-11800H (8C/16T), torch/BLAS threads capped at 4",
        "os": platform.platform(),
        "python": platform.python_version(),
    }
    if gpu:
        import torch

        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
            info["torch"] = torch.__version__
    return info


def write_report(name: str, payload: dict[str, Any], reports_dir: Path = REPORTS_DIR) -> Path:
    reports_dir.mkdir(parents=True, exist_ok=True)
    payload = {"_generated": dt.datetime.now().isoformat(timespec="seconds"), **payload}
    out = reports_dir / f"{name}.json"
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return out


def read_report(name: str, reports_dir: Path = REPORTS_DIR) -> dict[str, Any]:
    return json.loads((reports_dir / f"{name}.json").read_text(encoding="utf-8"))
