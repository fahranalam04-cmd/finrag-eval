"""Index-building stages: BM25 (tuned on dev), dense embeddings (GPU), Qdrant collections."""

from __future__ import annotations

import time

import numpy as np

from finrag.bm25 import BM25Index
from finrag.config import EMBEDDERS, Paths
from finrag.data import Passage
from finrag.dense import QdrantDenseIndex
from finrag.experiments.common import (
    REPORT_SPLIT,
    TUNING_SPLIT,
    corpus,
    hardware,
    qrels,
    queries,
    read_report,
    require_tuning_split,
    run_path,
    save_runs,
    write_report,
)
from finrag.metrics import evaluate_run
from finrag.types import ranked_ids

BM25_GRID = [(k1, b) for k1 in (0.9, 1.2, 1.5) for b in (0.4, 0.75, 0.9)]
CANDIDATES = 100
QDRANT_COLLECTION = "fiqa"


def passage_text(p: Passage) -> str:
    return f"{p.title}. {p.text}" if p.title else p.text


def tune_and_build_bm25(paths: Paths, split: str = TUNING_SPLIT) -> dict:
    """Grid-search BM25 k1/b on dev nDCG@10, persist the winner, cache dev + test top-100 runs."""
    require_tuning_split(split)
    docs = corpus(paths)
    ids, texts = [p.doc_id for p in docs], [passage_text(p) for p in docs]
    probe = BM25Index()
    t0 = time.perf_counter()
    tokens = probe.tokenize(texts)
    tok_s = time.perf_counter() - t0
    dev_q = queries(split, paths)
    grid = []
    for k1, b in BM25_GRID:
        idx = BM25Index(k1, b).build(ids, texts, tokens=tokens)
        runs = idx.search_batch([q.text for q in dev_q], k=CANDIDATES)
        ev = evaluate_run(
            {q.query_id: ranked_ids(r) for q, r in zip(dev_q, runs, strict=True)}, qrels(split, paths)
        )
        grid.append({"k1": k1, "b": b, **{m: round(v, 4) for m, v in ev.mean.items()}})
        print(f"[bm25] k1={k1} b={b} dev nDCG@10={ev.mean['ndcg@10']:.4f}", flush=True)
    best = max(grid, key=lambda r: (r["ndcg@10"], r["recall@100"]))
    t0 = time.perf_counter()
    final = BM25Index(best["k1"], best["b"]).build(ids, texts, tokens=tokens)
    build_s = time.perf_counter() - t0
    final.save(paths.bm25_dir)
    for s in (TUNING_SPLIT, REPORT_SPLIT):
        qs = queries(s, paths)
        res = final.search_batch([q.text for q in qs], k=CANDIDATES)
        save_runs(run_path(paths, "bm25", s), {q.query_id: r for q, r in zip(qs, res, strict=True)})
    report = {
        "split": split,
        "n_queries": len(dev_q),
        "selection_metric": "ndcg@10",
        "grid": grid,
        "selected": {"k1": best["k1"], "b": best["b"]},
        "tokenize_seconds": round(tok_s, 1),
        "index_build_seconds": round(build_s, 1),
        "hardware": hardware(),
    }
    write_report("dev_bm25_grid", report)
    return report


def embed(paths: Paths, names: list[str], batch_size: int = 128) -> dict:
    """GPU stage: encode the corpus and dev/test queries for each embedder; save float32 .npy files."""
    import torch

    from finrag.embedders import SentenceTransformerEmbedder

    docs = corpus(paths)
    texts = [passage_text(p) for p in docs]
    try:  # merge with earlier runs so embedders can be (re)built one at a time
        stats = read_report("embedding_stats")
        stats.pop("_generated", None)
    except FileNotFoundError:
        stats = {}
    stats |= {"hardware": hardware(gpu=True), "batch_size": batch_size, "fp16": True, "max_seq_length": 512}
    for name in names:
        cfg = EMBEDDERS[name]
        emb = SentenceTransformerEmbedder(
            cfg["hf_id"], name, cfg["query_prefix"], cfg["passage_prefix"], batch_size=batch_size
        )
        torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        vecs = emb.encode_passages(texts, show_progress=True)
        secs = time.perf_counter() - t0
        out = paths.embeddings_npy(name)
        out.parent.mkdir(parents=True, exist_ok=True)
        np.save(out, vecs)
        q_secs = {}
        for split in (TUNING_SPLIT, REPORT_SPLIT):
            qs = queries(split, paths)
            t1 = time.perf_counter()
            np.save(paths.query_embeddings_npy(name, split), emb.encode_queries([q.text for q in qs]))
            q_secs[split] = round(time.perf_counter() - t1, 2)
        stats[name] = {
            "hf_id": cfg["hf_id"],
            "dim": emb.dim,
            "n_passages": len(texts),
            "corpus_encode_seconds": round(secs, 1),
            "passages_per_second": round(len(texts) / secs, 1),
            "query_encode_seconds": q_secs,
            "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
        }
        print(f"[embed] {name}: {len(texts)} passages in {secs:.1f}s", flush=True)
        del emb
        torch.cuda.empty_cache()
        write_report("embedding_stats", stats)
    return stats


def build_qdrant(paths: Paths, names: list[str]) -> dict:
    """CPU stage: load each embedder's vectors into its own embedded Qdrant collection, cache top-100 runs."""
    docs = corpus(paths)
    out: dict = {"mode": "qdrant-client embedded/local (exact search)", "hardware": hardware()}
    for name in names:
        vecs = np.load(paths.embeddings_npy(name))
        idx = QdrantDenseIndex.local(paths.qdrant_dir / name, QDRANT_COLLECTION)
        t0 = time.perf_counter()
        idx.build(docs, vecs)
        build_s = time.perf_counter() - t0
        n = idx.count()
        timings = {}
        for split in (TUNING_SPLIT, REPORT_SPLIT):
            qs = queries(split, paths)
            qv = np.load(paths.query_embeddings_npy(name, split))
            t1 = time.perf_counter()
            res = idx.search_batch(qv, k=CANDIDATES)
            timings[split] = round((time.perf_counter() - t1) / len(qs) * 1e3, 2)
            save_runs(
                run_path(paths, f"dense-{name}", split), {q.query_id: r for q, r in zip(qs, res, strict=True)}
            )
        idx.close()
        out[name] = {"n_points": n, "upsert_seconds": round(build_s, 1), "batch_search_ms_per_query": timings}
        print(f"[qdrant] {name}: {n} points, upsert {build_s:.1f}s", flush=True)
    write_report("qdrant_build", out)
    return out
