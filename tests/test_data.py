"""Cleaning, qrels parsing and split leakage guards."""

from __future__ import annotations

import pytest

from finrag.data import assert_disjoint_splits, clean_text, load_qrels, split_queries
from finrag.experiments.common import require_tuning_split


def test_clean_text_unescapes_and_normalises():
    raw = "S&amp;P 500&nbsp;index<br/>funds​  are   cheap.\n\nReally &quot;cheap&quot;"
    assert clean_text(raw) == 'S&P 500 index funds are cheap. Really "cheap"'
    assert clean_text(None) == ""
    assert clean_text("x < y and y > z") == "x < y and y > z"  # comparison signs survive


def test_fixture_passages_are_cleaned(passages):
    assert "S&P 500" in passages[0].text and "&amp;" not in passages[0].text


def test_load_qrels(tmp_path):
    p = tmp_path / "q.tsv"
    p.write_text("query-id\tcorpus-id\tscore\n1\td1\t1\n1\td2\t0\n2\td3\t2\n", encoding="utf-8")
    assert load_qrels(p) == {"1": {"d1": 1}, "2": {"d3": 2}}  # zero-score rows dropped
    bad = tmp_path / "bad.tsv"
    bad.write_text("a\tb\tc\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_qrels(bad)


def test_split_queries_and_disjointness(fixture_queries, fixture_qrels):
    dev = split_queries(fixture_queries, fixture_qrels["dev"])
    assert [q.query_id for q in dev] == ["1", "2", "3"]
    assert_disjoint_splits(fixture_qrels)
    with pytest.raises(ValueError, match="shared"):
        assert_disjoint_splits({"dev": {"1": {"a": 1}}, "test": {"1": {"b": 1}}})
    with pytest.raises(KeyError):
        split_queries(fixture_queries, {"999": {"a": 1}})


def test_tuning_guard_refuses_test_split():
    require_tuning_split("dev")
    with pytest.raises(PermissionError):
        require_tuning_split("test")
