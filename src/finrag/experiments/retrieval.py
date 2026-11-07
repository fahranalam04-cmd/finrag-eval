"""Retrieval experiments: dev tuning (embedder, RRF, reranker/depth) and the one-shot test evaluation.

Protocol: every choice is made on dev nDCG@10 (greedy, stage by stage); `evaluate_test` only reads the
frozen choices from reports/dev_tuning.json and scores all variants on the 648 test queries.
"""

from __future__ import annotations

import csv
import json
import time
from collections.abc import Mapping
from pathlib import Path

from finrag.config import EMBEDDERS, REPORTS_DIR, RERANKERS, Paths
from finrag.experiments.common import (
    REPORT_SPLIT,
    TUNING_SPLIT,
    corpus_by_id,
    hardware,
    load_runs,
    qrels,
    queries,
    read_report,
    require_tuning_split,
    run_path,
    write_report,
)
from finrag.metrics import DEFAULT_METRICS, bootstrap_ci, evaluate_run, paired_bootstrap
from finrag.rerank import apply_rerank
from finrag.retriever import RetrieverConfig
from finrag.types import Ranked, ranked_ids

RRF_K_GRID = (1, 5, 10, 20, 40, 60, 100)
DENSE_WEIGHT_GRID = (1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 10.0)
RERANK_DEPTH_GRID = (5, 10, 20, 30, 50)
RERANK_FUSION_K_GRID = (5, 20, 60)  # rank fusion of first-stage and CE orders
RERANK_WEIGHT_GRID = (0.25, 0.5, 1.0)
RERANK_KEYS = ("rerank_depth", "rerank_fusion_k", "rerank_weight")
SELECT = "ndcg@10"

Runs = dict[str, Ranked]
CEScores = dict[str, dict[str, float]]


def _ids(runs: Mapping[str, Ranked]) -> dict[str, list[str]]:
    return {q: ranked_ids(r) for q, r in runs.items()}


def _mean(ev_mean: Mapping[str, float]) -> dict[str, float]:
    return {m: round(v, 4) for m, v in ev_mean.items()}


def fused_runs(bm25: Runs, dense: Runs, cfg: RetrieverConfig) -> Runs:
    qids = sorted(set(bm25) | set(dense))
    return {q: cfg.fuse(ranked_ids(bm25.get(q, [])), ranked_ids(dense.get(q, []))) for q in qids}


def reranked_runs(
    fused: Runs, ce: CEScores, depth: int, fusion_k: int | None = None, weight: float = 1.0
) -> Runs:
    out = {}
    for q, cands in fused.items():
        head = cands[:depth]
        out[q] = apply_rerank(cands, [ce[q][d] for d, _ in head], depth, fusion_k, weight)
    return out


def ce_path(paths: Paths, reranker: str, embedder: str, split: str) -> Path:
    return paths.runs_dir / f"ce-{reranker}.{embedder}.{split}.json"


# ------------------------------------------------------------------------------------------ dev tuning


def tune_first_stage(paths: Paths, split: str = TUNING_SPLIT) -> dict:
    require_tuning_split(split)
    qr = qrels(split, paths)
    bm25 = load_runs(run_path(paths, "bm25", split))
    report: dict = {"split": split, "n_queries": len(qr), "selection_metric": SELECT}
    report["bm25"] = _mean(evaluate_run(_ids(bm25), qr).mean)
    report["bm25_params"] = read_report("dev_bm25_grid")["selected"]

    dense_all = {}
    report["dense"] = {}
    for name in EMBEDDERS:
        p = run_path(paths, f"dense-{name}", split)
        if p.exists():
            dense_all[name] = load_runs(p)
            report["dense"][name] = _mean(evaluate_run(_ids(dense_all[name]), qr).mean)
    emb = max(report["dense"], key=lambda n: report["dense"][n][SELECT])
    report["selected_embedder"] = emb

    grid = []
    for k in RRF_K_GRID:
        for w in DENSE_WEIGHT_GRID:
            cfg = RetrieverConfig(rrf_k=k, dense_weight=w)
            ev = evaluate_run(_ids(fused_runs(bm25, dense_all[emb], cfg)), qr)
            grid.append({"rrf_k": k, "dense_weight": w, **_mean(ev.mean)})
    best = max(grid, key=lambda r: (r[SELECT], r["recall@100"]))
    report["rrf_grid"] = grid
    report["selected_fusion"] = {"rrf_k": best["rrf_k"], "dense_weight": best["dense_weight"]}
    report["hybrid_selected"] = {m: best[m] for m in DEFAULT_METRICS}
    report["hybrid_plain_rrf60"] = next(
        {m: r[m] for m in DEFAULT_METRICS} for r in grid if r["rrf_k"] == 60 and r["dense_weight"] == 1.0
    )
    write_report("dev_tuning", report)
    print(
        json.dumps(
            {k: report[k] for k in ("bm25", "dense", "selected_embedder", "selected_fusion")}, indent=1
        )
    )
    return report


def candidate_pool(paths: Paths, split: str, depth: int) -> dict[str, list[str]]:
    """Top-`depth` ids of the dev-tuned fused list: everything a reranker at depth <= `depth` can see.
    Scoring this pool once lets the dev grid sweep rerank depth without re-running the cross-encoder."""
    tuning = read_report("dev_tuning")
    emb, cfg = tuning["selected_embedder"], RetrieverConfig(**tuning["selected_fusion"])
    fused = fused_runs(
        load_runs(run_path(paths, "bm25", split)), load_runs(run_path(paths, f"dense-{emb}", split)), cfg
    )
    return {q: ranked_ids(r)[:depth] for q, r in fused.items()}


def score_candidates(paths: Paths, split: str, rerankers: list[str], batch_size: int = 64) -> dict:
    """GPU stage: cross-encoder scores for the fused candidates. Dev scores the deepest grid depth;
    test scores only each reranker's frozen dev-best depth."""
    import torch

    from finrag.rerank import CrossEncoderReranker

    tuning = read_report("dev_tuning")
    emb = tuning["selected_embedder"]
    qtext = {q.query_id: q.text for q in queries(split, paths)}
    docs = corpus_by_id(paths)
    stats = {}
    for name in rerankers:
        depth = (
            max(RERANK_DEPTH_GRID)
            if split == TUNING_SPLIT
            else tuning["best_rerank_per_reranker"][name]["rerank_depth"]
        )
        pool = candidate_pool(paths, split, depth)
        pairs_idx = [(q, d) for q, ds in pool.items() for d in ds]
        # length-sorted batching to minimise padding; scores are mapped back by (q, d)
        order = sorted(range(len(pairs_idx)), key=lambda i: len(docs[pairs_idx[i][1]].text))
        rr = CrossEncoderReranker(RERANKERS[name], name, batch_size=batch_size)
        t0 = time.perf_counter()
        scores = rr.score_pairs([(qtext[pairs_idx[i][0]], docs[pairs_idx[i][1]].text) for i in order])
        torch.cuda.synchronize()
        secs = time.perf_counter() - t0
        ce: CEScores = {q: {} for q in pool}
        for i, s in zip(order, scores.tolist(), strict=True):
            q, d = pairs_idx[i]
            ce[q][d] = s
        out = ce_path(paths, name, emb, split)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(ce), encoding="utf-8")
        stats[name] = {
            "hf_id": RERANKERS[name],
            "depth": depth,
            "n_pairs": len(pairs_idx),
            "seconds": round(secs, 1),
            "pairs_per_second": round(len(pairs_idx) / secs, 1),
            "batch_size": batch_size,
            "fp16": True,
            "max_length": 512,
        }
        print(f"[rerank-score] {name} {split}: {len(pairs_idx)} pairs in {secs:.1f}s", flush=True)
        del rr
        torch.cuda.empty_cache()
    try:
        prev = read_report("rerank_scoring")
    except FileNotFoundError:
        prev = {}
    prev[split] = {**prev.get(split, {}), "embedder": emb, "hardware": hardware(gpu=True), **stats}
    write_report("rerank_scoring", {k: v for k, v in prev.items() if not k.startswith("_")})
    return stats


def tune_rerank(paths: Paths, split: str = TUNING_SPLIT) -> dict:
    require_tuning_split(split)
    report = read_report("dev_tuning")
    report.pop("_generated", None)
    emb = report["selected_embedder"]
    cfg = RetrieverConfig(**report["selected_fusion"])
    qr = qrels(split, paths)
    fused = fused_runs(
        load_runs(run_path(paths, "bm25", split)), load_runs(run_path(paths, f"dense-{emb}", split)), cfg
    )
    grid = []
    for name in RERANKERS:
        p = ce_path(paths, name, emb, split)
        if not p.exists():
            continue
        ce = json.loads(p.read_text(encoding="utf-8"))
        settings = [(d, None, 1.0) for d in RERANK_DEPTH_GRID] + [
            (d, k, w) for d in RERANK_DEPTH_GRID for k in RERANK_FUSION_K_GRID for w in RERANK_WEIGHT_GRID
        ]
        for depth, fk, w in settings:
            ev = evaluate_run(_ids(reranked_runs(fused, ce, depth, fk, w)), qr)
            grid.append(
                {
                    "reranker": name,
                    "rerank_depth": depth,
                    "rerank_fusion_k": fk,
                    "rerank_weight": w,
                    **_mean(ev.mean),
                }
            )
    report["rerank_grid"] = grid
    report["best_rerank_per_reranker"] = {
        n: max((r for r in grid if r["reranker"] == n), key=lambda r: r[SELECT])
        for n in sorted({r["reranker"] for r in grid})
    }
    report["best_pure_rerank_per_reranker"] = {
        n: max(
            (r for r in grid if r["reranker"] == n and r["rerank_fusion_k"] is None), key=lambda r: r[SELECT]
        )
        for n in sorted({r["reranker"] for r in grid})
    }
    best = max(grid, key=lambda r: r[SELECT])
    # the reranker must earn its place: it is only switched on if it beats the tuned hybrid on dev
    if best[SELECT] > report["hybrid_selected"][SELECT]:
        report["selected_rerank"] = {"reranker": best["reranker"], **{k: best[k] for k in RERANK_KEYS}}
        report["hybrid_rerank_selected"] = {m: best[m] for m in DEFAULT_METRICS}
    else:
        report["selected_rerank"] = None
        report["hybrid_rerank_selected"] = None
    write_report("dev_tuning", report)
    print(json.dumps({k: report[k] for k in ("selected_rerank", "hybrid_rerank_selected")}, indent=1))
    return report


def selected_config(tuning: Mapping | None = None) -> tuple[str, RetrieverConfig, str | None]:
    """(embedder, retriever config, reranker name or None) frozen from the dev tuning report."""
    t = tuning or read_report("dev_tuning")
    sel = t.get("selected_rerank")
    rr = {k: sel[k] for k in RERANK_KEYS} if sel else {}
    return (
        t["selected_embedder"],
        RetrieverConfig(**t["selected_fusion"], **rr),
        sel["reranker"] if sel else None,
    )


def variant_runs(paths: Paths, split: str) -> tuple[dict[str, Runs], dict]:
    """All retrieval variants for a split, built from cached first-stage runs + CE scores and the frozen
    dev selections. Returns (variant -> runs, description of the selected pipeline)."""
    tuning = read_report("dev_tuning")
    emb, cfg, rr_sel = selected_config(tuning)
    bm25 = load_runs(run_path(paths, "bm25", split))
    variants: dict[str, Runs] = {"bm25": bm25}
    for name in EMBEDDERS:
        p = run_path(paths, f"dense-{name}", split)
        if p.exists():
            variants[f"dense:{name}"] = load_runs(p)
    plain = fused_runs(bm25, variants[f"dense:{emb}"], RetrieverConfig(rrf_k=60, dense_weight=1.0))
    fused = fused_runs(bm25, variants[f"dense:{emb}"], cfg)
    variants["hybrid:rrf60"] = plain
    variants["hybrid:tuned"] = fused
    for name, best in tuning["best_rerank_per_reranker"].items():
        p = ce_path(paths, name, emb, split)
        if p.exists():
            ce = json.loads(p.read_text(encoding="utf-8"))
            variants[f"hybrid+rerank:{name}"] = reranked_runs(fused, ce, *(best[k] for k in RERANK_KEYS))
            pure = tuning["best_pure_rerank_per_reranker"][name]
            if pure["rerank_depth"] <= best["rerank_depth"]:  # CE scores cover this depth
                variants[f"hybrid+pure-rerank:{name}"] = reranked_runs(fused, ce, pure["rerank_depth"])
    selected = {
        "embedder": emb,
        "rrf_k": cfg.rrf_k,
        "dense_weight": cfg.dense_weight,
        "reranker": rr_sel,
        "rerank_depth": cfg.rerank_depth if rr_sel else None,
        "rerank_fusion_k": cfg.rerank_fusion_k if rr_sel else None,
        "rerank_weight": cfg.rerank_weight if rr_sel else None,
        "bm25": tuning["bm25_params"],
        "candidates_per_retriever": cfg.candidates,
        "variant": f"hybrid+rerank:{rr_sel}" if rr_sel else "hybrid:tuned",
    }
    return variants, selected


# --------------------------------------------------------------------------------------- test (once)


def _ir_measures_crosscheck(
    runs: dict[str, list[str]], qr: Mapping[str, Mapping[str, int]], ours: Mapping
) -> dict:
    import ir_measures
    from ir_measures import RR, R, nDCG

    measures = {"ndcg@10": nDCG @ 10, "recall@10": R @ 10, "recall@100": R @ 100, "mrr@10": RR @ 10}
    qrels_l = [ir_measures.Qrel(q, d, int(r)) for q, ds in qr.items() for d, r in ds.items()]
    # scores strictly decreasing with rank so trec_eval's score sort reproduces our order exactly
    run_l = [
        ir_measures.ScoredDoc(q, d, float(len(ds) - i)) for q, ds in runs.items() for i, d in enumerate(ds)
    ]
    agg = ir_measures.calc_aggregate(list(measures.values()), qrels_l, run_l)
    theirs = {m: float(agg[meas]) for m, meas in measures.items()}
    return {"ir_measures": theirs, "max_abs_diff": max(abs(theirs[m] - ours[m]) for m in measures)}


def evaluate_split(paths: Paths, split: str = REPORT_SPLIT, n_boot: int = 10_000) -> dict:
    variants, selected = variant_runs(paths, split)
    qr = qrels(split, paths)
    results: dict = {}
    per_query: dict[str, dict[str, float]] = {}
    for name, runs in variants.items():
        ids = _ids(runs)
        ev = evaluate_run(ids, qr)
        row: dict = {"n_queries": ev.n_queries, **_mean(ev.mean)}
        row["ci95"] = {
            m: [round(x, 4) for x in bootstrap_ci(list(ev.per_query[m].values()), n_boot)] for m in ev.mean
        }
        row["crosscheck"] = _ir_measures_crosscheck(ids, qr, ev.mean)
        results[name] = row
        per_query[name] = ev.per_query[SELECT]
    qids = sorted(per_query["bm25"])
    sel = selected["variant"]
    dense_sel = f"dense:{selected['embedder']}"

    def cmp(a: str, b: str) -> dict:
        r = paired_bootstrap([per_query[a][q] for q in qids], [per_query[b][q] for q in qids], n_boot)
        base = results[a][SELECT]
        return {k: round(v, 4) for k, v in r.items()} | {
            "relative_lift": round(r["diff"] / base, 4) if base else None
        }

    pairs = [("bm25", sel), ("bm25", dense_sel), (dense_sel, "hybrid:tuned"), (dense_sel, "hybrid:rrf60")]
    pairs += [("hybrid:tuned", v) for v in variants if v.startswith("hybrid+")]
    comparisons = {f"{b} vs {a}": cmp(a, b) for a, b in dict.fromkeys(pairs) if a != b}
    report = {
        "split": split,
        "n_queries": len(qr),
        "metrics_note": "means over all split queries; ci95 = percentile bootstrap over queries (10k draws)",
        "selected_pipeline": selected,
        "variants": results,
        "paired_bootstrap_ndcg@10": comparisons,
        "hardware": hardware(),
    }
    name = "retrieval_test" if split == REPORT_SPLIT else f"retrieval_{split}"
    write_report(name, report)
    with open(REPORTS_DIR / f"{name}_per_query_ndcg10.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["query_id", *variants])
        for q in qids:
            w.writerow([q, *(round(per_query[v][q], 4) for v in variants)])
    return report
