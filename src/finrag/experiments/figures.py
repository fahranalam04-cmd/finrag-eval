"""README figures, rendered only from reports/*.json (no recomputation)."""

from __future__ import annotations

from finrag.config import FIGURES_DIR
from finrag.experiments.common import read_report

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, BLUE_LIGHT, ORANGE, AQUA, NEUTRAL = "#2a78d6", "#86b6ef", "#eb6834", "#1baf7a", "#a3a29c"

LABELS = {
    "bm25": "BM25 (tuned k1/b)",
    "dense:bge-small": "Dense: bge-small",
    "dense:bge-base": "Dense: bge-base",
    "hybrid:rrf60": "Hybrid RRF (k=60, untuned)",
    "hybrid:tuned": "Hybrid RRF (dev-tuned)",
    "hybrid+rerank:minilm-l6": "Hybrid + MiniLM-L6 (rank fusion)",
    "hybrid+rerank:bge-reranker-base": "Hybrid + bge-reranker-base (rank fusion)",
    "hybrid+pure-rerank:minilm-l6": "Hybrid + MiniLM-L6 (pure rerank)",
    "hybrid+pure-rerank:bge-reranker-base": "Hybrid + bge-reranker-base (pure rerank)",
}


def _style(ax) -> None:
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def retrieval_bars() -> None:
    import matplotlib.pyplot as plt

    r = read_report("retrieval_test")
    sel = r["selected_pipeline"]["variant"]
    rows = sorted(r["variants"].items(), key=lambda kv: kv[1]["ndcg@10"])
    fig, ax = plt.subplots(figsize=(7.2, 3.6), dpi=150, facecolor=SURFACE)
    _style(ax)
    ax.grid(axis="y", visible=False)
    for i, (name, row) in enumerate(rows):
        v, (lo, hi) = row["ndcg@10"], row["ci95"]["ndcg@10"]
        color = BLUE if name == sel else (NEUTRAL if name == "bm25" else BLUE_LIGHT)
        ax.barh(i, v, height=0.62, color=color, edgecolor=SURFACE, linewidth=2)
        ax.errorbar(v, i, xerr=[[v - lo], [hi - v]], fmt="none", ecolor=INK2, elinewidth=1, capsize=2.5)
        ax.text(hi + 0.006, i, f"{v:.3f}", va="center", fontsize=8.5, color=INK)
    ax.set_yticks(
        range(len(rows)), [LABELS.get(n, n) + ("  (selected)" if n == sel else "") for n, _ in rows]
    )
    ax.set_xlabel("nDCG@10 on FiQA test (648 queries), 95% bootstrap CI", color=INK2, fontsize=9)
    ax.set_xlim(0, max(row["ci95"]["ndcg@10"][1] for _, row in rows) * 1.12)
    ax.set_title("Retrieval quality by variant", loc="left", color=INK, fontsize=11)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "retrieval_ndcg10_test.png", facecolor=SURFACE)
    plt.close(fig)


def latency_vs_quality() -> None:
    import matplotlib.pyplot as plt

    r = read_report("retrieval_test")
    lat = read_report("latency")
    sel = r["selected_pipeline"]
    quality = {
        "bm25": r["variants"]["bm25"]["ndcg@10"],
        "dense": r["variants"][f"dense:{sel['embedder']}"]["ndcg@10"],
        "hybrid": r["variants"]["hybrid:tuned"]["ndcg@10"],
        "hybrid_rerank": r["variants"][sel["variant"]]["ndcg@10"],
    }
    names = {"bm25": "BM25", "dense": "Dense", "hybrid": "Hybrid", "hybrid_rerank": "Hybrid+rerank"}
    fig, ax = plt.subplots(figsize=(7.2, 3.6), dpi=150, facecolor=SURFACE)
    _style(ax)
    for dev, color, label in (("cuda", BLUE, "models on GPU"), ("cpu", ORANGE, "CPU only")):
        if dev not in lat:
            continue
        xs = [lat[dev]["modes"][m]["stages_ms"]["total"]["p50"] for m in quality]
        ax.plot(xs, list(quality.values()), "-", color=color, linewidth=2, alpha=0.5)
        ax.scatter(
            xs,
            list(quality.values()),
            s=46,
            color=color,
            edgecolor=SURFACE,
            linewidth=2,
            zorder=3,
            label=label,
        )
        for x, (m, q) in zip(xs, quality.items(), strict=True):
            ax.annotate(
                f"{names[m]}\n{x:.0f} ms",
                (x, q),
                textcoords="offset points",
                xytext=(6, -14),
                fontsize=7.5,
                color=INK2,
            )
    ax.set_xscale("log")
    ax.set_xlabel(
        "p50 end-to-end retrieval latency per query, ms (log scale, batch size 1)", color=INK2, fontsize=9
    )
    ax.set_ylabel("test nDCG@10", color=INK2, fontsize=9)
    ax.set_title(
        f"Latency vs quality (rerank depth {sel['rerank_depth']})", loc="left", color=INK, fontsize=11
    )
    ax.legend(frameon=False, fontsize=8.5, labelcolor=INK2, loc="lower right")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "latency_vs_quality.png", facecolor=SURFACE)
    plt.close(fig)


def generation_tradeoff() -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    g = read_report("generation_eval")
    cfgs = [c for c in ("selected_top3", "selected_top5", "bm25_top5") if c in g["configs"]]
    labels = {
        "selected_top3": "selected pipeline, top-3",
        "selected_top5": "selected pipeline, top-5",
        "bm25_top5": "BM25, top-5",
    }
    metrics = [
        ("faithfulness_sentence", "Faithfulness\n(sentences entailed)"),
        ("citation_precision_qrels", "Citation precision\n(vs qrels)"),
        ("abstention_rate", "Abstention rate"),
    ]
    colors = [BLUE, ORANGE, AQUA]
    fig, ax = plt.subplots(figsize=(7.2, 3.4), dpi=150, facecolor=SURFACE)
    _style(ax)
    ax.grid(axis="x", visible=False)
    x = np.arange(len(metrics))
    w = 0.8 / len(cfgs)
    for i, c in enumerate(cfgs):
        vals = [g["configs"][c][m] or 0.0 for m, _ in metrics]
        pos = x - 0.4 + w / 2 + i * w
        ax.bar(pos, vals, width=w, color=colors[i], edgecolor=SURFACE, linewidth=2, label=labels[c])
        for p, v in zip(pos, vals, strict=True):
            ax.text(p, v + 0.012, f"{v:.2f}", ha="center", fontsize=7.5, color=INK)
    ax.set_xticks(x, [m[1] for m in metrics])
    ax.set_ylim(0, 1)
    n = g["configs"][cfgs[0]]["n_queries"]
    ax.set_title(f"Generation (Qwen2.5-1.5B-Instruct, {n} test queries)", loc="left", color=INK, fontsize=11)
    ax.legend(frameon=False, fontsize=8.5, labelcolor=INK2, ncols=3, loc="upper right")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "generation_tradeoff.png", facecolor=SURFACE)
    plt.close(fig)


def make_all() -> None:
    import matplotlib

    matplotlib.use("Agg")
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    retrieval_bars()
    latency_vs_quality()
    generation_tradeoff()
