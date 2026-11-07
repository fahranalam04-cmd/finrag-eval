"""`finrag` command line: one subcommand per pipeline stage, plus `all` to run them end to end.

Stages marked [GPU] use CUDA; on a shared machine wrap them in a GPU lock (see scripts/reproduce.sh).
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable

from finrag.config import EMBEDDERS, RERANKERS, Paths
from finrag.experiments.common import set_cpu_threads, write_report


def _prepare(a: argparse.Namespace, p: Paths) -> None:
    from finrag.data import download_fiqa

    write_report("data_stats", download_fiqa(p))


def _fetch_models(a: argparse.Namespace, p: Paths) -> None:
    """Pre-download pinned weights (safetensors + configs only) so GPU-lock time is not spent on I/O."""
    from huggingface_hub import snapshot_download

    from finrag.config import MODEL_REVISIONS

    for hf_id, rev in MODEL_REVISIONS.items():
        path = snapshot_download(
            hf_id,
            revision=rev,
            allow_patterns=[
                "*.json",
                "*.safetensors",
                "*.txt",
                "*.model",
                "tokenizer*",
                "1_Pooling/*",
                "*.py",
            ],
        )
        print(f"[fetch] {hf_id}@{rev[:8]} -> {path}", flush=True)


def _bm25(a: argparse.Namespace, p: Paths) -> None:
    from finrag.experiments.index import tune_and_build_bm25

    tune_and_build_bm25(p)


def _embed(a: argparse.Namespace, p: Paths) -> None:
    from finrag.experiments.index import embed

    embed(p, a.embedders, batch_size=a.batch_size)


def _qdrant(a: argparse.Namespace, p: Paths) -> None:
    from finrag.experiments.index import build_qdrant

    build_qdrant(p, a.embedders)


def _tune_first(a: argparse.Namespace, p: Paths) -> None:
    from finrag.experiments.retrieval import tune_first_stage

    tune_first_stage(p)


def _score(a: argparse.Namespace, p: Paths) -> None:
    from finrag.experiments.retrieval import score_candidates

    score_candidates(p, a.split, a.rerankers)


def _tune_rerank(a: argparse.Namespace, p: Paths) -> None:
    from finrag.experiments.retrieval import tune_rerank

    tune_rerank(p)


def _evaluate(a: argparse.Namespace, p: Paths) -> None:
    from finrag.experiments.retrieval import evaluate_split

    r = evaluate_split(p, a.split)
    print(
        json.dumps(
            {
                k: {m: v[m] for m in ("ndcg@10", "recall@10", "recall@100", "mrr@10")}
                for k, v in r["variants"].items()
            },
            indent=1,
        )
    )


def _latency(a: argparse.Namespace, p: Paths) -> None:
    from finrag.experiments.latency import run

    run(p, a.device, a.n_queries, a.n_rerank_queries)


def _generate(a: argparse.Namespace, p: Paths) -> None:
    from finrag.experiments.generation_eval import generate

    generate(p, a.split, a.n, a.seed, batch_size=a.batch_size)


def _judge(a: argparse.Namespace, p: Paths) -> None:
    from finrag.experiments.generation_eval import judge

    judge(p, a.split)


def _figures(a: argparse.Namespace, p: Paths) -> None:
    from finrag.experiments.figures import make_all

    make_all()


def _all(a: argparse.Namespace, p: Paths) -> None:
    ns = argparse.Namespace(
        embedders=list(EMBEDDERS),
        rerankers=list(RERANKERS),
        batch_size=128,
        split="dev",
        device="cuda",
        n_queries=200,
        n_rerank_queries=None,
        n=200,
        seed=13,
    )
    steps: list[tuple[str, Callable[[argparse.Namespace, Paths], None], dict]] = [
        ("prepare", _prepare, {}),
        ("fetch-models", _fetch_models, {}),
        ("bm25", _bm25, {}),
        ("embed", _embed, {}),
        ("qdrant", _qdrant, {}),
        ("tune-first-stage", _tune_first, {}),
        ("score-rerank dev", _score, {"split": "dev"}),
        ("tune-rerank", _tune_rerank, {}),
        ("score-rerank test", _score, {"split": "test"}),
        ("evaluate test", _evaluate, {"split": "test"}),
        ("latency cuda", _latency, {"device": "cuda"}),
        ("latency cpu", _latency, {"device": "cpu", "n_queries": 100, "n_rerank_queries": 30}),
        ("generate test", _generate, {"split": "test", "batch_size": 8}),
        ("judge test", _judge, {"split": "test"}),
        ("figures", _figures, {}),
    ]
    for name, fn, kw in steps:
        print(f"==> {name}", flush=True)
        fn(argparse.Namespace(**{**vars(ns), **kw}), p)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="finrag", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("prepare", help="download + clean FiQA (pinned revisions)").set_defaults(fn=_prepare)
    sub.add_parser("fetch-models", help="pre-download pinned model weights").set_defaults(fn=_fetch_models)
    sub.add_parser("bm25", help="tune BM25 k1/b on dev, build index, cache runs").set_defaults(fn=_bm25)
    s = sub.add_parser("embed", help="[GPU] encode corpus + dev/test queries")
    s.add_argument("--embedders", nargs="+", default=list(EMBEDDERS), choices=list(EMBEDDERS))
    s.add_argument("--batch-size", type=int, default=128)
    s.set_defaults(fn=_embed)
    s = sub.add_parser("qdrant", help="load vectors into embedded Qdrant, cache dense runs")
    s.add_argument("--embedders", nargs="+", default=list(EMBEDDERS), choices=list(EMBEDDERS))
    s.set_defaults(fn=_qdrant)
    sub.add_parser("tune-first-stage", help="dev: pick embedder + RRF k/weight").set_defaults(fn=_tune_first)
    s = sub.add_parser("score-rerank", help="[GPU] cross-encoder scores for the candidate pool")
    s.add_argument("--split", choices=["dev", "test"], required=True)
    s.add_argument("--rerankers", nargs="+", default=list(RERANKERS), choices=list(RERANKERS))
    s.set_defaults(fn=_score)
    sub.add_parser("tune-rerank", help="dev: pick reranker + depth").set_defaults(fn=_tune_rerank)
    s = sub.add_parser("evaluate", help="score all variants with frozen dev choices")
    s.add_argument("--split", choices=["dev", "test"], default="test")
    s.set_defaults(fn=_evaluate)
    s = sub.add_parser("latency", help="[GPU if cuda] per-stage single-query latency")
    s.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    s.add_argument("--n-queries", type=int, default=200)
    s.add_argument("--n-rerank-queries", type=int, default=None)
    s.set_defaults(fn=_latency)
    s = sub.add_parser("generate", help="[GPU] grounded answers for a fixed query sample")
    s.add_argument("--split", choices=["dev", "test"], default="test")
    s.add_argument("--n", type=int, default=200)
    s.add_argument("--seed", type=int, default=13)
    s.add_argument("--batch-size", type=int, default=8)
    s.set_defaults(fn=_generate)
    s = sub.add_parser("judge", help="[GPU] NLI faithfulness + citation metrics")
    s.add_argument("--split", choices=["dev", "test"], default="test")
    s.set_defaults(fn=_judge)
    sub.add_parser("figures", help="render reports/figures/*.png").set_defaults(fn=_figures)
    sub.add_parser("all", help="run every stage in order").set_defaults(fn=_all)
    return ap


def main(argv: list[str] | None = None) -> int:
    set_cpu_threads(4)
    args = build_parser().parse_args(argv)
    args.fn(args, Paths())
    return 0


if __name__ == "__main__":
    sys.exit(main())
