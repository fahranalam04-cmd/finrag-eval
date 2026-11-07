"""Citation parsing, abstention, sentence splitting and the answer-level metric aggregation."""

from __future__ import annotations

import pytest

from finrag.faithfulness import OverlapNLIScorer, aggregate, judge_answer, split_sentences
from finrag.generation import (
    ABSTAIN_PHRASE,
    ContextPassage,
    FakeGenerator,
    build_messages,
    is_abstention,
    parse_citations,
    strip_citations,
    truncate_words,
)


def test_parse_citations_formats():
    a = "Fees are low [doc101]. Taxes apply [doc102, doc103][doc101] and [Doc 104]. See [1] and [note]."
    assert parse_citations(a) == ["101", "102", "103", "104"]
    assert parse_citations("no citations here") == []


def test_strip_citations_keeps_other_brackets():
    assert strip_citations("Fees are low [doc101].") == "Fees are low."
    assert strip_citations("Rates [see note] rose [doc1][doc2] fast.") == "Rates [see note] rose fast."


def test_abstention_detection():
    assert is_abstention("not enough information")
    assert is_abstention("Not enough information.")
    assert is_abstention("There is not enough information in the passages to answer.")
    assert not is_abstention("Index funds are cheap [doc101].")
    # a long uncited answer that ends with a hedge is an (unsupported) answer, not an abstention
    long_hedge = "Index funds are cheap and diversified and they track an index closely over time. " * 2
    assert not is_abstention(long_hedge + "Otherwise there is not enough information.")
    # says the phrase but still makes a cited claim -> counted as an answer
    assert not is_abstention("Not enough information on fees, but they are passive [doc101].")


def test_split_sentences_keeps_trailing_citations_with_their_sentence():
    a = "Index funds are cheap. [doc101] They track the U.S. market closely [doc102]. Yes."
    s = split_sentences(a)
    assert s == ["Index funds are cheap [doc101].", "They track the U.S. market closely [doc102]. Yes."]
    assert split_sentences("- First point is here [doc1].\n- Second point is here [doc2].") == [
        "First point is here [doc1].",
        "Second point is here [doc2].",
    ]


def test_prompt_contains_ids_and_truncates():
    msgs = build_messages("Why?", [ContextPassage("7", "word " * 500)], max_words=10)
    assert msgs[0]["role"] == "system" and ABSTAIN_PHRASE in msgs[0]["content"]
    assert "[doc7] " in msgs[-1]["content"] and msgs[-1]["content"].count("word") == 10
    assert msgs[-2]["content"] == ABSTAIN_PHRASE  # the few-shot abstention demo precedes the question
    assert truncate_words("a b c", 5) == "a b c"


def test_fake_generator_cites_top_passage_and_abstains_without_context():
    out = FakeGenerator().generate_batch(
        [("q", [ContextPassage("9", "Bonds pay coupons. More text.")]), ("q", [])]
    )
    assert out[0].answer == "Bonds pay coupons. [doc9]" and out[0].cited_ids == ["9"] and not out[0].abstained
    assert out[1].abstained


CTX = {
    "101": "Index funds have low expense ratios because they are passively managed.",
    "102": "Qualified Roth IRA withdrawals in retirement are tax free.",
    "103": "Bonds pay periodic coupon interest.",
}
QRELS = {"101": 1, "102": 1}


def judge(answer: str, abstained: bool = False):
    return judge_answer("q", answer, abstained, CTX, QRELS, OverlapNLIScorer(), max_words=180)


def test_judge_answer_citation_validity_precision_and_faithfulness():
    j = judge(
        "Index funds have low expense ratios [doc101]. "
        "Roth IRA withdrawals are tax free [doc999]. "
        "Bonds always double in value [doc103]."
    )
    assert j.cited_ids == ["101", "999", "103"]
    assert j.valid_cited == ["101", "103"]  # doc999 was never in the prompt
    assert j.relevant_cited == ["101"]
    assert j.relevant_in_context == 2
    supported = [s.p_entail_cited >= 0.5 for s in j.sentences]
    # s1 entailed by its cited passage; s2's only own citation is invalid -> falls back to the answer's
    # valid citations (101, 103), which do not entail it; s3 is a hallucination
    assert supported == [True, False, False]
    # context faithfulness ignores citations: s2 *is* supported by passage 102
    assert [s.p_entail_context >= 0.5 for s in j.sentences] == [True, True, False]


def test_aggregate_metrics():
    js = [
        judge("Index funds have low expense ratios [doc101]."),
        judge("Bonds pay periodic coupon interest [doc103]. Bonds always double [doc103]."),
        judge("not enough information", abstained=True),
        judge("Roth IRA withdrawals are tax free."),  # no citation at all -> unsupported
    ]
    agg = aggregate(js)
    assert agg["n_queries"] == 4 and agg["n_answered"] == 3
    assert agg["abstention_rate"] == pytest.approx(0.25)
    assert agg["answers_with_citation_rate"] == pytest.approx(2 / 3)
    assert agg["citation_validity"] == pytest.approx(1.0)
    assert agg["citation_precision_qrels"] == pytest.approx(1 / 2)  # 101 relevant, 103 not
    assert agg["n_sentences"] == 4
    assert agg["faithfulness_sentence"] == pytest.approx(2 / 4)
    assert agg["faithfulness_answer_macro"] == pytest.approx((1 + 0.5 + 0) / 3)
    assert agg["fully_faithful_answer_rate"] == pytest.approx(1 / 3)
    assert agg["context_faithfulness_sentence"] == pytest.approx(3 / 4)
    assert agg["context_precision_qrels"] == pytest.approx(8 / 12)
