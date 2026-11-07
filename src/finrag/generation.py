"""Grounded answer generation with inline citations, behind a backend-agnostic `Generator` protocol.

Only the local Hugging Face backend is implemented/run; another backend (vLLM, an API, ...) only has
to implement `generate_batch`.
"""

from __future__ import annotations

import re
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Protocol

ABSTAIN_PHRASE = "not enough information"
DEFAULT_MAX_WORDS = 180  # per-passage truncation in the prompt (FiQA passage p50 = 90 words, p95 = 368)

SYSTEM_PROMPT = (
    "You are a careful financial question-answering assistant. Answer the question using ONLY the "
    "passages provided.\n"
    "Rules:\n"
    "1. After every sentence, cite the passage(s) that support it using their ids in square brackets, "
    "for example [doc12345]. Only cite ids that appear in the passages.\n"
    "2. Do not use outside knowledge and do not invent facts, numbers or ids.\n"
    f"3. If the passages do not contain enough information to answer, reply with exactly: {ABSTAIN_PHRASE}\n"
    "4. Be concise: at most 4 sentences."
)


@dataclass(frozen=True)
class ContextPassage:
    doc_id: str
    text: str


@dataclass
class GenerationResult:
    answer: str
    cited_ids: list[str]
    abstained: bool
    prompt_tokens: int = 0
    completion_tokens: int = 0
    meta: dict = field(default_factory=dict)


class Generator(Protocol):
    name: str

    def generate_batch(
        self, items: Sequence[tuple[str, Sequence[ContextPassage]]]
    ) -> list[GenerationResult]: ...


def truncate_words(text: str, max_words: int) -> str:
    words = text.split()
    return text if len(words) <= max_words else " ".join(words[:max_words]) + " ..."


def format_context(passages: Sequence[ContextPassage], max_words: int = DEFAULT_MAX_WORDS) -> str:
    return "\n\n".join(f"[doc{p.doc_id}] {truncate_words(p.text, max_words)}" for p in passages)


def build_messages(
    question: str, passages: Sequence[ContextPassage], max_words: int = DEFAULT_MAX_WORDS
) -> list[dict[str, str]]:
    user = (
        f"Passages:\n{format_context(passages, max_words)}\n\n"
        f"Question: {question}\n\n"
        "Answer in at most 4 sentences, citing passages like [doc12345] after each sentence, "
        f"or reply exactly '{ABSTAIN_PHRASE}'."
    )
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]


_BRACKET = re.compile(r"\[([^\[\]]{1,200})\]")
_DOC_ID = re.compile(r"\bdoc\s*[-_#:]?\s*(\d[\w-]*)", re.IGNORECASE)  # FiQA ids are numeric


def parse_citations(answer: str) -> list[str]:
    """Ids cited as [doc123], [doc1, doc2] or [doc1][doc2]; ordered, de-duplicated."""
    out: list[str] = []
    for inner in _BRACKET.findall(answer):
        for d in _DOC_ID.findall(inner):
            if d not in out:
                out.append(d)
    return out


def strip_citations(text: str) -> str:
    t = _BRACKET.sub(lambda m: "" if _DOC_ID.search(m.group(1)) else m.group(0), text)
    t = re.sub(r"\s+([.,;:!?])", r"\1", t)
    return re.sub(r"\s+", " ", t).strip()


def is_abstention(answer: str, cited_ids: Sequence[str] | None = None) -> bool:
    """Abstained = the answer contains the abstain phrase and cites nothing (an answer that says the
    phrase but still makes cited claims is treated as an answer)."""
    cited = parse_citations(answer) if cited_ids is None else cited_ids
    norm = re.sub(r"[^a-z ]", " ", answer.lower())
    return ABSTAIN_PHRASE in " ".join(norm.split()) and not cited


def make_result(answer: str, **kw) -> GenerationResult:
    answer = answer.strip()
    cited = parse_citations(answer)
    return GenerationResult(answer=answer, cited_ids=cited, abstained=is_abstention(answer, cited), **kw)


class FakeGenerator:
    """Deterministic extractive stand-in for tests: first sentence of the top passage + its citation."""

    name = "fake-extractive"

    def generate_batch(self, items: Sequence[tuple[str, Sequence[ContextPassage]]]) -> list[GenerationResult]:
        out = []
        for _question, passages in items:
            if not passages:
                out.append(make_result(ABSTAIN_PHRASE))
                continue
            p = passages[0]
            first = re.split(r"(?<=[.!?])\s+", p.text.strip())[0]
            out.append(make_result(f"{first} [doc{p.doc_id}]"))
        return out


class HFLocalGenerator:
    """Local open-weight instruct model (default Qwen2.5-1.5B-Instruct, fp16), greedy decoding."""

    def __init__(
        self,
        hf_id: str,
        device: str = "cuda",
        max_new_tokens: int = 220,
        batch_size: int = 8,
        max_words: int = DEFAULT_MAX_WORDS,
    ) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        from finrag.config import revision

        self._torch = torch
        self.tok = AutoTokenizer.from_pretrained(hf_id, revision=revision(hf_id))
        self.tok.padding_side = "left"
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            hf_id, revision=revision(hf_id), dtype=torch.float16 if device == "cuda" else torch.float32
        ).to(device)
        self.model.eval()
        self.name, self.hf_id, self.device = hf_id.split("/")[-1], hf_id, device
        self.max_new_tokens, self.batch_size, self.max_words = max_new_tokens, batch_size, max_words

    def _prompt(self, question: str, passages: Sequence[ContextPassage]) -> str:
        return self.tok.apply_chat_template(
            build_messages(question, passages, self.max_words), tokenize=False, add_generation_prompt=True
        )

    def generate_batch(self, items: Sequence[tuple[str, Sequence[ContextPassage]]]) -> list[GenerationResult]:
        torch = self._torch
        prompts = [self._prompt(q, ps) for q, ps in items]
        order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]))  # length-bucketed batches
        results: list[GenerationResult | None] = [None] * len(prompts)
        for start in range(0, len(order), self.batch_size):
            idx = order[start : start + self.batch_size]
            enc = self.tok([prompts[i] for i in idx], return_tensors="pt", padding=True).to(self.device)
            t0 = time.perf_counter()
            with torch.inference_mode():
                out = self.model.generate(
                    **enc,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,
                    temperature=None,
                    top_p=None,
                    top_k=None,
                    pad_token_id=self.tok.pad_token_id,
                )
            if self.device == "cuda":
                torch.cuda.synchronize()
            dt = time.perf_counter() - t0
            new = out[:, enc["input_ids"].shape[1] :]
            texts = self.tok.batch_decode(new, skip_special_tokens=True)
            for row, i in enumerate(idx):
                n_new = int((new[row] != self.tok.pad_token_id).sum())
                results[i] = make_result(
                    texts[row],
                    prompt_tokens=int(enc["attention_mask"][row].sum()),
                    completion_tokens=n_new,
                    meta={"batch_seconds": dt, "batch_size": len(idx)},
                )
        return [r for r in results if r is not None]

    def close(self) -> None:
        del self.model
        if self.device == "cuda":
            self._torch.cuda.empty_cache()
