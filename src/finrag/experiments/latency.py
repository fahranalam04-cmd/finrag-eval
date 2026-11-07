"""Per-stage online latency of the serving retriever (one query at a time, batch size 1)."""

from __future__ import annotations

import random
import time

import numpy as np

from finrag.config import Paths
from finrag.experiments.common import REPORT_SPLIT, hardware, queries, read_report, write_report
from finrag.experiments.retrieval import selected_config
from finrag.rag import load_retriever
from finrag.retriever import MODES


def _summary(xs: list[float]) -> dict[str, float]:
    a = np.asarray(xs)
    return {
        "p50": round(float(np.percentile(a, 50)), 2),
        "p95": round(float(np.percentile(a, 95)), 2),
        "mean": round(float(a.mean()), 2),
    }


def measure(
    paths: Paths,
    device: str,
    n_queries: int = 200,
    n_rerank_queries: int | None = None,
    warmup: int = 10,
    seed: int = 0,
) -> dict:
    if device == "cpu":
        import torch

        torch.set_num_threads(4)
    t0 = time.perf_counter()
    retr = load_retriever(paths, device)
    load_s = time.perf_counter() - t0
    qs = queries(REPORT_SPLIT, paths)
    rng = random.Random(seed)
    sample = rng.sample(qs, n_queries)
    out: dict = {
        "device_for_models": device,
        "cpu_only_stages": ["bm25", "dense_search (Qdrant embedded, exact)", "fusion", "fetch_text"],
        "batch_size": 1,
        "k_returned": 10,
        "load_seconds": round(load_s, 1),
        "modes": {},
    }
    for mode in MODES:
        n = n_queries if (mode != "hybrid_rerank" or n_rerank_queries is None) else n_rerank_queries
        for q in sample[:warmup]:
            retr.search(q.text, mode, k=10)
        stages: dict[str, list[float]] = {}
        for q in sample[:n]:
            res = retr.search(q.text, mode, k=10)
            for s, v in res.timings_ms.items():
                stages.setdefault(s, []).append(v)
        out["modes"][mode] = {"n_queries": n, "stages_ms": {s: _summary(v) for s, v in stages.items()}}
        print(
            f"[latency:{device}] {mode}: total p50={out['modes'][mode]['stages_ms']['total']['p50']} ms",
            flush=True,
        )
    return out


def run(paths: Paths, device: str, n_queries: int, n_rerank_queries: int | None = None) -> dict:
    res = measure(paths, device, n_queries, n_rerank_queries)
    try:
        report = read_report("latency")
        report.pop("_generated", None)
    except FileNotFoundError:
        report = {}
    emb, cfg, rr = selected_config()
    report["pipeline"] = {
        "embedder": emb,
        "reranker": rr,
        "rerank_depth": cfg.rerank_depth,
        "candidates_per_retriever": cfg.candidates,
        "split": REPORT_SPLIT,
    }
    report[device] = res | {"hardware": hardware(gpu=device == "cuda")}
    write_report("latency", report)
    return res
