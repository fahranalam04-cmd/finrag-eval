"""Answer-level evaluation: citation validity / precision, NLI faithfulness, abstention.

Definitions (all computed per answer, then aggregated):
- citation validity  : cited id was one of the passages actually placed in the prompt.
- citation precision : valid cited id is judged relevant in the FiQA qrels (a lower bound: qrels are
                       sparse, ~2.6 judged passages per query, so unjudged != irrelevant).
- faithfulness       : share of answer sentences entailed (NLI p_entail >= threshold) by at least one
                       passage the sentence cites; a sentence without its own citation is checked
                       against the passages cited anywhere in the answer; an answer with no valid
                       citation has every sentence unsupported.
- context faithfulness: same, but any prompt passage may support the sentence (ignores citations).
- abstention         : see generation.is_abstention.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from typing import Protocol

import numpy as np

from finrag.generation import parse_citations, strip_citations, truncate_words

_ABBREV = re.compile(
    r"\b(U\.S|U\.K|e\.g|i\.e|etc|vs|Mr|Mrs|Dr|Inc|Ltd|Co|Corp|No|approx|St)\.", re.IGNORECASE
)
_CITE_AFTER_PUNCT = re.compile(r"([.!?])((?:\s*\[[^\[\]]*doc[^\[\]]*\])+)", re.IGNORECASE)
_BOUNDARY = re.compile(r"(?<=[.!?])\s+(?=[\"'(\[A-Z0-9$])")
_BULLET = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s+")


def split_sentences(answer: str, min_words: int = 3) -> list[str]:
    """Split an answer into claim-bearing sentences, keeping each citation with the sentence it ends.

    "A is true. [doc1] B holds [doc2]." -> ["A is true [doc1].", "B holds [doc2]."]
    Fragments with fewer than `min_words` words (after removing citations) are merged into the
    previous sentence.
    """
    sents: list[str] = []
    for line in answer.splitlines():
        line = _BULLET.sub("", line).strip()
        if not line:
            continue
        line = _CITE_AFTER_PUNCT.sub(lambda m: m.group(2) + m.group(1), line)
        line = _ABBREV.sub(lambda m: m.group(1) + "<DOT>", line)
        for part in _BOUNDARY.split(line):
            part = part.replace("<DOT>", ".").strip()
            if not part:
                continue
            if sents and len(strip_citations(part).split()) < min_words:
                sents[-1] = f"{sents[-1]} {part}"
            else:
                sents.append(part)
    # a short first fragment can only be merged forward
    if len(sents) > 1 and len(strip_citations(sents[0]).split()) < min_words:
        sents[1] = f"{sents[0]} {sents[1]}"
        sents = sents[1:]
    return sents


class NLIScorer(Protocol):
    def entailment_probs(self, pairs: Sequence[tuple[str, str]]) -> np.ndarray:
        """P(premise entails hypothesis) for each (premise, hypothesis) pair."""
        ...


class HFNLIScorer:
    def __init__(self, hf_id: str, device: str = "cuda", batch_size: int = 32, max_length: int = 512) -> None:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        from finrag.config import revision

        self._torch = torch
        self.tok = AutoTokenizer.from_pretrained(hf_id, revision=revision(hf_id))
        self.model = (
            AutoModelForSequenceClassification.from_pretrained(hf_id, revision=revision(hf_id))
            .to(device)
            .eval()
        )
        labels = {v.lower(): int(k) for k, v in self.model.config.id2label.items()}
        self.entail_idx = labels["entailment"]
        self.device, self.batch_size, self.max_length = device, batch_size, max_length
        self._cache: dict[tuple[str, str], float] = {}

    def entailment_probs(self, pairs: Sequence[tuple[str, str]]) -> np.ndarray:
        todo = list(dict.fromkeys(p for p in pairs if p not in self._cache))
        torch = self._torch
        for start in range(0, len(todo), self.batch_size):
            chunk = todo[start : start + self.batch_size]
            enc = self.tok(
                [p for p, _ in chunk],
                [h for _, h in chunk],
                truncation="only_first",
                max_length=self.max_length,
                padding=True,
                return_tensors="pt",
            ).to(self.device)
            with torch.inference_mode():
                probs = torch.softmax(self.model(**enc).logits.float(), dim=-1)[:, self.entail_idx]
            for pair, pr in zip(chunk, probs.cpu().tolist(), strict=True):
                self._cache[pair] = float(pr)
        return np.asarray([self._cache[p] for p in pairs], dtype=np.float32)

    def close(self) -> None:
        del self.model
        if self.device == "cuda":
            self._torch.cuda.empty_cache()


class OverlapNLIScorer:
    """Model-free test stand-in: 'entailed' iff every content word of the hypothesis is in the premise."""

    _tok = re.compile(r"[a-z0-9]{3,}")

    def entailment_probs(self, pairs: Sequence[tuple[str, str]]) -> np.ndarray:
        out = []
        for prem, hyp in pairs:
            h, p = set(self._tok.findall(hyp.lower())), set(self._tok.findall(prem.lower()))
            out.append(1.0 if h and h <= p else 0.0)
        return np.asarray(out, dtype=np.float32)


@dataclass
class SentenceJudgment:
    sentence: str
    cited_ids: list[str]
    own_citation: bool
    p_entail_cited: float
    p_entail_context: float


@dataclass
class AnswerJudgment:
    query_id: str
    abstained: bool
    cited_ids: list[str]
    valid_cited: list[str]
    relevant_cited: list[str]
    context_ids: list[str]
    relevant_in_context: int
    sentences: list[SentenceJudgment] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def judge_answer(
    query_id: str,
    answer: str,
    abstained: bool,
    context: Mapping[str, str],
    qrels_for_query: Mapping[str, int],
    nli: NLIScorer,
    max_words: int,
) -> AnswerJudgment:
    """`context` maps prompt passage ids -> text (in prompt order)."""
    cited = parse_citations(answer)
    valid = [d for d in cited if d in context]
    j = AnswerJudgment(
        query_id=query_id,
        abstained=abstained,
        cited_ids=cited,
        valid_cited=valid,
        relevant_cited=[d for d in valid if qrels_for_query.get(d, 0) > 0],
        context_ids=list(context),
        relevant_in_context=sum(qrels_for_query.get(d, 0) > 0 for d in context),
    )
    if abstained:
        return j
    premises = {d: truncate_words(t, max_words) for d, t in context.items()}
    sents = split_sentences(answer)
    plan: list[tuple[str, list[str], bool, str]] = []
    pairs: list[tuple[str, str]] = []
    for s in sents:
        own = [d for d in parse_citations(s) if d in context]
        use = own or valid
        hyp = strip_citations(s)
        plan.append((s, use, bool(own), hyp))
        pairs.extend((premises[d], hyp) for d in context)  # context covers cited ⊆ context
    probs = nli.entailment_probs(pairs) if pairs else np.zeros(0)
    ids = list(context)
    for k, (s, use, own, _hyp) in enumerate(plan):
        row = probs[k * len(ids) : (k + 1) * len(ids)]
        by_id = dict(zip(ids, row.tolist(), strict=True))
        j.sentences.append(
            SentenceJudgment(
                sentence=s,
                cited_ids=use,
                own_citation=own,
                p_entail_cited=max((by_id[d] for d in use), default=0.0),
                p_entail_context=max(by_id.values(), default=0.0),
            )
        )
    return j


def _safe_div(a: float, b: float) -> float | None:
    return a / b if b else None


def aggregate(judgments: Sequence[AnswerJudgment], threshold: float = 0.5) -> dict:
    n = len(judgments)
    answered = [j for j in judgments if not j.abstained]
    n_cited = sum(len(j.cited_ids) for j in answered)
    n_valid = sum(len(j.valid_cited) for j in answered)
    sents = [s for j in answered for s in j.sentences]
    per_answer_faith = [
        float(np.mean([s.p_entail_cited >= threshold for s in j.sentences])) for j in answered if j.sentences
    ]
    with_rel = [j for j in judgments if j.relevant_in_context > 0]
    without_rel = [j for j in judgments if j.relevant_in_context == 0]
    return {
        "n_queries": n,
        "n_answered": len(answered),
        "abstention_rate": _safe_div(n - len(answered), n),
        "abstention_rate_context_has_relevant": _safe_div(sum(j.abstained for j in with_rel), len(with_rel)),
        "abstention_rate_context_no_relevant": _safe_div(
            sum(j.abstained for j in without_rel), len(without_rel)
        ),
        "n_context_has_relevant": len(with_rel),
        "answers_with_citation_rate": _safe_div(sum(bool(j.cited_ids) for j in answered), len(answered)),
        "citation_validity": _safe_div(n_valid, n_cited),
        "answers_all_citations_valid_rate": _safe_div(
            sum(bool(j.cited_ids) and len(j.valid_cited) == len(j.cited_ids) for j in answered), len(answered)
        ),
        "citation_precision_qrels": _safe_div(sum(len(j.relevant_cited) for j in answered), n_valid),
        "context_precision_qrels": _safe_div(
            sum(j.relevant_in_context for j in judgments), sum(len(j.context_ids) for j in judgments)
        ),
        "n_sentences": len(sents),
        "sentence_own_citation_rate": _safe_div(sum(s.own_citation for s in sents), len(sents)),
        "faithfulness_sentence": _safe_div(sum(s.p_entail_cited >= threshold for s in sents), len(sents)),
        "faithfulness_answer_macro": float(np.mean(per_answer_faith)) if per_answer_faith else None,
        "fully_faithful_answer_rate": _safe_div(sum(f == 1.0 for f in per_answer_faith), len(answered)),
        "context_faithfulness_sentence": _safe_div(
            sum(s.p_entail_context >= threshold for s in sents), len(sents)
        ),
        "nli_threshold": threshold,
    }


def per_answer_faithfulness(judgments: Sequence[AnswerJudgment], threshold: float = 0.5) -> dict[str, float]:
    """query_id -> share of entailed sentences, for answered queries (used for paired bootstrap)."""
    return {
        j.query_id: float(np.mean([s.p_entail_cited >= threshold for s in j.sentences]))
        for j in judgments
        if not j.abstained and j.sentences
    }
