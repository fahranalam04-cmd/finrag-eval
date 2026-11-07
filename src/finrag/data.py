"""FiQA-2018 (BEIR) loading, cleaning and split handling.

The raw parquet files are fetched once from the Hugging Face Hub at pinned revisions and converted to
small cleaned JSONL / TSV files; everything downstream reads those, so the core package needs no pyarrow.
"""

from __future__ import annotations

import csv
import html
import json
import re
import unicodedata
from collections import defaultdict
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

from finrag.config import (
    FIQA_QRELS_REPO,
    FIQA_QRELS_REVISION,
    FIQA_REPO,
    FIQA_REVISION,
    SPLITS,
    Paths,
)

Qrels = dict[str, dict[str, int]]  # query_id -> {doc_id: relevance}

_ZERO_WIDTH = re.compile(r"[​-‏  ﻿]")
_WS = re.compile(r"\s+")
_HTML_TAG = re.compile(r"</?(?:p|br|b|i|a|div|span|ul|ol|li)\b[^>]*>", re.IGNORECASE)


@dataclass(frozen=True)
class Passage:
    doc_id: str
    text: str
    title: str = ""


@dataclass(frozen=True)
class Query:
    query_id: str
    text: str


def clean_text(text: str | None) -> str:
    """Normalise a FiQA passage/query: unescape HTML entities, drop stray tags and zero-width chars,
    NFKC-normalise and collapse whitespace. Content words are never changed."""
    if not text:
        return ""
    t = html.unescape(text)
    t = _HTML_TAG.sub(" ", t)
    t = _ZERO_WIDTH.sub("", t)
    t = unicodedata.normalize("NFKC", t)
    return _WS.sub(" ", t).strip()


def read_jsonl(path: Path) -> Iterator[dict]:
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def write_jsonl(path: Path, rows: Iterable[dict]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    return n


def load_corpus(path: Path) -> list[Passage]:
    return [Passage(r["doc_id"], r["text"], r.get("title", "")) for r in read_jsonl(path)]


def load_queries(path: Path) -> dict[str, Query]:
    return {r["query_id"]: Query(r["query_id"], r["text"]) for r in read_jsonl(path)}


def load_qrels(path: Path) -> Qrels:
    """Read a BEIR qrels TSV (header: query-id, corpus-id, score). Zero-score rows are dropped."""
    qrels: Qrels = defaultdict(dict)
    with open(path, encoding="utf-8", newline="") as f:
        reader = csv.reader(f, delimiter="\t")
        header = next(reader)
        if header[:2] != ["query-id", "corpus-id"]:
            raise ValueError(f"unexpected qrels header {header}")
        for row in reader:
            qid, did, score = row[0], row[1], int(row[2])
            if score > 0:
                qrels[qid][did] = score
    return dict(qrels)


def split_queries(queries: dict[str, Query], qrels: Qrels) -> list[Query]:
    """Queries belonging to a split = those with at least one judged-relevant passage in its qrels,
    in a deterministic (sorted-id) order."""
    missing = [q for q in qrels if q not in queries]
    if missing:
        raise KeyError(f"{len(missing)} qrels query ids not in queries file, e.g. {missing[:3]}")
    return [queries[q] for q in sorted(qrels, key=_id_key)]


def _id_key(x: str) -> tuple[int, str]:
    return (int(x), x) if x.isdigit() else (1 << 62, x)


def assert_disjoint_splits(split_qrels: dict[str, Qrels]) -> None:
    """Leakage guard: no query may appear in two splits."""
    names = list(split_qrels)
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            overlap = set(split_qrels[a]) & set(split_qrels[b])
            if overlap:
                raise ValueError(f"{len(overlap)} queries shared by {a} and {b}, e.g. {sorted(overlap)[:3]}")


# --------------------------------------------------------------------------------------------- download


def download_fiqa(paths: Paths) -> dict:
    """Fetch pinned parquet/TSV files, clean them and write JSONL. Returns dataset statistics."""
    import pyarrow.parquet as pq  # heavy optional deps: only needed here
    from huggingface_hub import hf_hub_download

    raw = paths.raw_dir
    raw.mkdir(parents=True, exist_ok=True)
    corpus_pq = hf_hub_download(
        FIQA_REPO,
        "corpus/corpus-00000-of-00001.parquet",
        repo_type="dataset",
        revision=FIQA_REVISION,
        local_dir=raw,
    )
    queries_pq = hf_hub_download(
        FIQA_REPO,
        "queries/queries-00000-of-00001.parquet",
        repo_type="dataset",
        revision=FIQA_REVISION,
        local_dir=raw,
    )
    qrel_files = {
        s: hf_hub_download(
            FIQA_QRELS_REPO, f"{s}.tsv", repo_type="dataset", revision=FIQA_QRELS_REVISION, local_dir=raw
        )
        for s in SPLITS
    }

    corpus_tbl = pq.read_table(corpus_pq).to_pylist()
    n_empty = 0
    rows = []
    for r in corpus_tbl:
        text = clean_text(r["text"])
        n_empty += text == ""
        rows.append({"doc_id": str(r["_id"]), "title": clean_text(r.get("title")), "text": text})
    if len({r["doc_id"] for r in rows}) != len(rows):
        raise ValueError("duplicate doc ids in corpus")
    write_jsonl(paths.corpus_jsonl, rows)
    word_counts = sorted(len(r["text"].split()) for r in rows)

    q_rows = [
        {"query_id": str(r["_id"]), "text": clean_text(r["text"])}
        for r in pq.read_table(queries_pq).to_pylist()
    ]
    write_jsonl(paths.queries_jsonl, q_rows)

    stats: dict = {
        "source": {
            "corpus_and_queries": f"https://huggingface.co/datasets/{FIQA_REPO} @ {FIQA_REVISION}",
            "qrels": f"https://huggingface.co/datasets/{FIQA_QRELS_REPO} @ {FIQA_QRELS_REVISION}",
            "license": "cc-by-sa-4.0 (per both dataset cards)",
        },
        "n_passages": len(rows),
        "n_empty_passages_after_cleaning": n_empty,
        "passage_words": {
            "mean": round(sum(word_counts) / len(word_counts), 1),
            "p50": word_counts[len(word_counts) // 2],
            "p95": word_counts[int(len(word_counts) * 0.95)],
            "max": word_counts[-1],
        },
        "n_queries_total": len(q_rows),
        "splits": {},
    }
    split_qrels = {}
    for s, f in qrel_files.items():
        dest = paths.qrels_tsv(s)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(Path(f).read_bytes())
        qr = load_qrels(dest)
        split_qrels[s] = qr
        n_pairs = sum(len(v) for v in qr.values())
        stats["splits"][s] = {
            "n_queries": len(qr),
            "n_judged_pairs": n_pairs,
            "mean_relevant_per_query": round(n_pairs / len(qr), 2),
            "relevance_values": sorted({v for d in qr.values() for v in d.values()}),
        }
    assert_disjoint_splits(split_qrels)
    corpus_ids = {r["doc_id"] for r in rows}
    stats["qrels_docs_missing_from_corpus"] = sum(
        d not in corpus_ids for qr in split_qrels.values() for docs in qr.values() for d in docs
    )
    return stats
