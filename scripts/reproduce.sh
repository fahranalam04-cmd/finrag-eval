#!/usr/bin/env bash
# Full pipeline, stage by stage. GPU stages are prefixed with $GPU_WRAP, e.g. on the shared dev laptop:
#   GPU_WRAP="python D:/ml-portfolio/gpu_lock.py --who finrag --" bash scripts/reproduce.sh
# On a normal machine leave GPU_WRAP empty (or just run `uv run finrag all`).
set -euo pipefail
GPU_WRAP=${GPU_WRAP:-}
uv run finrag prepare
uv run finrag fetch-models
uv run finrag bm25
$GPU_WRAP uv run finrag embed
uv run finrag qdrant
uv run finrag tune-first-stage
$GPU_WRAP uv run finrag score-rerank --split dev
uv run finrag tune-rerank
$GPU_WRAP uv run finrag score-rerank --split test
uv run finrag evaluate --split test
$GPU_WRAP uv run finrag latency --device cuda --n-queries 200
CUDA_VISIBLE_DEVICES="" uv run finrag latency --device cpu --n-queries 100 --n-rerank-queries 30
$GPU_WRAP uv run finrag generate --split test --n 200 --seed 13
$GPU_WRAP uv run finrag judge --split test
uv run finrag figures
