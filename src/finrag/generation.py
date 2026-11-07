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

_RULES_TAIL = (
    "2. Do not use outside knowledge and do not invent facts, numbers or ids.\n"
    f"3. If the passages do not contain enough information to answer, reply with exactly: {ABSTAIN_PHRASE}\n"
    "4. Be concise: at most 4 sentences."
)
_INTRO = (
    "You are a careful financial question-answering assistant. Answer the question using ONLY the "
    "passages provided.\nRules:\n"
)
SYSTEM_PROMPTS = {
    "v1": _INTRO + "1. After every sentence, cite the passage(s) that support it using their ids in square "
    "brackets, for example [doc12345]. Only cite ids that appear in the passages.\n" + _RULES_TAIL,
    "v2": _INTRO + "1. End every sentence with the id of the passage that supports it, in square brackets, "
    "exactly as written before that passage, e.g. [docID]. Only cite ids that appear in the passages.\n"
    + _RULES_TAIL,
}
V1_REMINDER = (
    "Answer in at most 4 sentences, citing passages like [doc12345] after each sentence, "
    f"or reply exactly '{ABSTAIN_PHRASE}'."
)

# Prompt selection on 24 dev queries (reports/generation_eval_dev_prompt_v{1,2}.json, selected_top5):
#   v1 = rules + an end-of-turn reminder with an example id: 45% of answers cite, sentence faithfulness 0.22
#        (one answer copied the example id [doc12345], which is then scored as an invalid citation).
#   v2 = one-shot citation demo + abstention demo, no end-of-turn reminder: only 17% of answers cite,
#        faithfulness 0.04. The 1.5B model follows the instruction nearest the question, not the demo.
# v1 is used for the test run; v2 is kept only so the dev comparison can be reproduced.
PROMPT_VERSION = "v1"
SYSTEM_PROMPT = SYSTEM_PROMPTS[PROMPT_VERSION]

# Demo ids have a leading zero, which never occurs in FiQA, so a copied demo id counts as invalid.
FEW_SHOT: list[dict[str, str]] = [
    {
        "role": "user",
        "content": "Passages:\n[doc0001] A Roth IRA is funded with after-tax money, so qualified withdrawals "
        "in retirement are tax free.\n\n[doc0002] Contributions to a traditional IRA may be deductible, "
        "but withdrawals in retirement are taxed as ordinary income.\n\nQuestion: Are Roth IRA "
        "withdrawals taxed?",
    },
    {
        "role": "assistant",
        "content": "Qualified Roth IRA withdrawals in retirement are tax free because the contributions were "
        "made with after-tax money [doc0001]. A traditional IRA works the other way: withdrawals are "
        "taxed as ordinary income [doc0002].",
    },
    {
        "role": "user",
        "content": "Passages:\n[doc0003] Bond prices fall when market interest rates rise.\n\n"
        "Question: Which credit card has the best travel rewards?",
    },
    {"role": "assistant", "content": ABSTAIN_PHRASE},
]


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
    question: str,
    passages: Sequence[ContextPassage],
    max_words: int = DEFAULT_MAX_WORDS,
    version: str = PROMPT_VERSION,
) -> list[dict[str, str]]:
    user = f"Passages:\n{format_context(passages, max_words)}\n\nQuestion: {question}"
    if version == "v1":
        return [
            {"role": "system", "content": SYSTEM_PROMPTS["v1"]},
            {"role": "user", "content": f"{user}\n\n{V1_REMINDER}"},
        ]
    if version == "v2":
        return [
            {"role": "system", "content": SYSTEM_PROMPTS["v2"]},
            *FEW_SHOT,
            {"role": "user", "content": user},
        ]
    raise ValueError(f"unknown prompt version {version!r}")


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


def is_abstention(answer: str, cited_ids: Sequence[str] | None = None, max_words: int = 15) -> bool:
    """Abstained = cites nothing, contains the abstain phrase, and either opens with it or is short.
    A long uncited answer that merely ends with a hedge is scored as an (unsupported) answer."""
    cited = parse_citations(answer) if cited_ids is None else cited_ids
    norm = " ".join(re.sub(r"[^a-z ]", " ", answer.lower()).split())
    if cited or ABSTAIN_PHRASE not in norm:
        return False
    return norm.startswith(ABSTAIN_PHRASE) or len(norm.split()) <= max_words


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
        vram_fraction: float | None = 0.8,
    ) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        from finrag.config import revision

        self._torch = torch
        if device == "cuda" and vram_fraction:
            # On Windows the NVIDIA driver's sysmem fallback lets cudaMalloc spill into host RAM instead of
            # failing, so PyTorch's caching allocator never frees its cache and the process grows in host
            # memory (observed: >5 GB RSS and a 2x slowdown). A hard cap makes the allocator free and retry.
            torch.cuda.set_per_process_memory_fraction(vram_fraction)
        self.tok = AutoTokenizer.from_pretrained(hf_id, revision=revision(hf_id))
        self.tok.padding_side = "left"
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        # device_map loads the weights straight onto the device. Loading on CPU and calling .to("cuda")
        # left a ~3 GB host copy resident (measured: 3.6 GB vs 0.8 GB RSS after load).
        self.model = AutoModelForCausalLM.from_pretrained(
            hf_id,
            revision=revision(hf_id),
            dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map=device,
        )
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
            out = out.cpu()
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
            del enc, out
            if self.device == "cuda":
                torch.cuda.empty_cache()  # length-bucketed batches all differ in shape; avoid fragmentation
        return [r for r in results if r is not None]

    def close(self) -> None:
        del self.model
        if self.device == "cuda":
            self._torch.cuda.empty_cache()
