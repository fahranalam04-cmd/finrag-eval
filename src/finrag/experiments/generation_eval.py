"""Generation experiments: grounded answers on a fixed query sample, then NLI / citation judging.

Contexts come from the *same* cached test runs that reports/retrieval_test.json scores, so generation
metrics and retrieval metrics describe one pipeline.
"""

from __future__ import annotations

import json
import random
import time
from collections.abc import Mapping

import numpy as np

from finrag.config import GENERATOR_HF_ID, NLI_HF_ID, REPORTS_DIR, Paths
from finrag.data import read_jsonl, write_jsonl
from finrag.experiments.common import corpus_by_id, hardware, qrels, queries, write_report
from finrag.experiments.retrieval import variant_runs
from finrag.faithfulness import aggregate, judge_answer, per_answer_faithfulness
from finrag.generation import DEFAULT_MAX_WORDS, PROMPT_VERSION, SYSTEM_PROMPT, ContextPassage
from finrag.metrics import bootstrap_ci, paired_bootstrap
from finrag.types import ranked_ids

# config name -> (retrieval variant key prefix, number of passages in the prompt)
# "selected" = the dev-selected retrieval pipeline (see reports/retrieval_test.json:selected_pipeline)
GEN_CONFIGS: dict[str, tuple[str, int]] = {
    "selected_top3": ("selected", 3),
    "selected_top5": ("selected", 5),
    "bm25_top5": ("bm25", 5),  # baseline retrieval feeding the same generator
}
PAIRED = [("selected_top3", "selected_top5"), ("bm25_top5", "selected_top5")]
SINGLE_CONFIG = "selected_top5"


def sample_query_ids(split: str, n: int, seed: int, paths: Paths) -> list[str]:
    ids = [q.query_id for q in queries(split, paths)]
    return sorted(random.Random(seed).sample(ids, min(n, len(ids))), key=lambda x: int(x))


def contexts(paths: Paths, split: str) -> dict[str, dict[str, list[str]]]:
    variants, selected = variant_runs(paths, split)
    out: dict[str, dict[str, list[str]]] = {}
    for cfg, (prefix, k) in GEN_CONFIGS.items():
        key = selected["variant"] if prefix == "selected" else prefix
        out[cfg] = {q: ranked_ids(r)[:k] for q, r in variants[key].items()}
    return out


def _gen_path(split: str):
    return REPORTS_DIR / f"generations_{split}.jsonl"


def generate(
    paths: Paths,
    split: str,
    n: int,
    seed: int,
    batch_size: int = 8,
    max_new_tokens: int = 220,
    n_single: int = 10,
    configs: list[str] | None = None,
) -> dict:
    """GPU stage. Writes reports/generations_<split>.jsonl and reports/generation_runs_<split>.json."""
    import torch

    from finrag.generation import HFLocalGenerator

    qids = sample_query_ids(split, n, seed, paths)
    qtext = {q.query_id: q.text for q in queries(split, paths)}
    docs = corpus_by_id(paths)
    ctx = contexts(paths, split)
    gen = HFLocalGenerator(GENERATOR_HF_ID, batch_size=batch_size, max_new_tokens=max_new_tokens)
    torch.cuda.reset_peak_memory_stats()
    records, timing = [], {}
    for cfg in configs or list(GEN_CONFIGS):
        items = [(qtext[q], [ContextPassage(d, docs[d].text) for d in ctx[cfg][q]]) for q in qids]
        t0 = time.perf_counter()
        res = gen.generate_batch(items)
        secs = time.perf_counter() - t0
        n_tok = sum(r.completion_tokens for r in res)
        timing[cfg] = {
            "seconds": round(secs, 1),
            "seconds_per_answer_amortised": round(secs / len(res), 3),
            "completion_tokens_per_second": round(n_tok / secs, 1),
            "mean_prompt_tokens": round(float(np.mean([r.prompt_tokens for r in res])), 1),
            "mean_completion_tokens": round(float(np.mean([r.completion_tokens for r in res])), 1),
            "hit_max_new_tokens": sum(r.completion_tokens >= max_new_tokens for r in res),
        }
        for q, r in zip(qids, res, strict=True):
            records.append(
                {
                    "config": cfg,
                    "query_id": q,
                    "question": qtext[q],
                    "context_ids": ctx[cfg][q],
                    "answer": r.answer,
                    "cited_ids": r.cited_ids,
                    "abstained": r.abstained,
                    "prompt_tokens": r.prompt_tokens,
                    "completion_tokens": r.completion_tokens,
                }
            )
        print(f"[generate] {cfg}: {len(res)} answers in {secs:.0f}s", flush=True)
    # online-style latency: one question at a time (batch size 1), selected config, first n_single queries
    gen.batch_size = 1
    single_ms = []
    for q in qids[:n_single]:
        item = (qtext[q], [ContextPassage(d, docs[d].text) for d in ctx[SINGLE_CONFIG][q]])
        t0 = time.perf_counter()
        gen.generate_batch([item])
        single_ms.append((time.perf_counter() - t0) * 1e3)
    gen.batch_size = batch_size
    peak = torch.cuda.max_memory_allocated() / 1e9
    gen.close()
    write_jsonl(_gen_path(split), records)
    report = {
        "split": split,
        "n_queries": len(qids),
        "sample_seed": seed,
        "query_ids": qids,
        "generator": {
            "hf_id": GENERATOR_HF_ID,
            "dtype": "float16",
            "decoding": "greedy",
            "max_new_tokens": max_new_tokens,
            "batch_size": batch_size,
            "passage_truncation_words": DEFAULT_MAX_WORDS,
        },
        "configs": {
            c: {"retrieval": GEN_CONFIGS[c][0], "k_context": GEN_CONFIGS[c][1]}
            for c in configs or GEN_CONFIGS
        },
        "prompt_version": PROMPT_VERSION,
        "system_prompt": SYSTEM_PROMPT,
        "timing": timing,
        "single_query_latency_ms": {
            "config": SINGLE_CONFIG,
            "batch_size": 1,
            "n_queries": len(single_ms),
            "p50": round(float(np.percentile(single_ms, 50)), 1) if single_ms else None,
            "p95": round(float(np.percentile(single_ms, 95)), 1) if single_ms else None,
        },
        "peak_vram_gb": round(peak, 2),
        "hardware": hardware(gpu=True),
    }
    write_report(f"generation_runs_{split}", report)
    return report


def _ci(xs: list[float]) -> list[float]:
    return [round(v, 4) for v in bootstrap_ci(xs)] if xs else []


def judge(paths: Paths, split: str, threshold: float = 0.5) -> dict:
    """GPU stage (NLI model). Writes reports/generation_eval[_<split>].json + per-answer judgments."""
    from finrag.faithfulness import HFNLIScorer

    records = list(read_jsonl(_gen_path(split)))
    docs = corpus_by_id(paths)
    qr = qrels(split, paths)
    nli = HFNLIScorer(NLI_HF_ID)
    t0 = time.perf_counter()
    by_cfg: dict[str, list] = {}
    for r in records:
        context: Mapping[str, str] = {d: docs[d].text for d in r["context_ids"]}
        j = judge_answer(
            r["query_id"],
            r["answer"],
            r["abstained"],
            context,
            qr.get(r["query_id"], {}),
            nli,
            DEFAULT_MAX_WORDS,
        )
        by_cfg.setdefault(r["config"], []).append(j)
    secs = time.perf_counter() - t0
    nli.close()
    results = {}
    for cfg, js in by_cfg.items():
        agg = aggregate(js, threshold)
        faith = list(per_answer_faithfulness(js, threshold).values())
        cit_prec = [
            len(j.relevant_cited) / len(j.valid_cited) for j in js if not j.abstained and j.valid_cited
        ]
        agg["ci95"] = {
            "faithfulness_answer_macro": _ci(faith),
            "citation_precision_answer_macro": _ci(cit_prec),
            "abstention_rate": _ci([float(j.abstained) for j in js]),
        }
        agg["citation_precision_answer_macro"] = round(float(np.mean(cit_prec)), 4) if cit_prec else None
        results[cfg] = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in agg.items()}
    paired = {}
    for a, b in [p for p in PAIRED if p[0] in by_cfg and p[1] in by_cfg]:
        fa, fb = per_answer_faithfulness(by_cfg[a], threshold), per_answer_faithfulness(by_cfg[b], threshold)
        both = sorted(set(fa) & set(fb))
        ab_a = {j.query_id: float(j.abstained) for j in by_cfg[a]}
        ab_b = {j.query_id: float(j.abstained) for j in by_cfg[b]}
        qs = sorted(ab_a)
        paired[f"{b} vs {a}"] = {
            "faithfulness_answer_macro": {"n_queries_answered_by_both": len(both)}
            | {
                k: round(v, 4)
                for k, v in paired_bootstrap([fa[q] for q in both], [fb[q] for q in both]).items()
            },
            "abstention_rate": {
                k: round(v, 4)
                for k, v in paired_bootstrap([ab_a[q] for q in qs], [ab_b[q] for q in qs]).items()
            },
        }
    write_jsonl(
        REPORTS_DIR / f"generation_judgments_{split}.jsonl",
        ({"config": cfg, **j.to_dict()} for cfg, js in by_cfg.items() for j in js),
    )
    report = {
        "split": split,
        "n_queries_per_config": {c: len(v) for c, v in by_cfg.items()},
        "nli_model": NLI_HF_ID,
        "nli_threshold": threshold,
        "definitions": "see finrag/faithfulness.py module docstring",
        "configs": results,
        "paired_bootstrap": paired,
        "judge_seconds": round(secs, 1),
        "hardware": hardware(gpu=True),
    }
    name = "generation_eval" if split == "test" else f"generation_eval_{split}"
    write_report(name, report)
    print(json.dumps({c: {k: v for k, v in r.items() if k != "ci95"} for c, r in results.items()}, indent=1))
    return report
