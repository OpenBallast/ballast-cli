from pathlib import Path

import pytest

from openballast.store import Store, candidate_spans, format_value, norm


def test_norm():
    assert norm("Douglas  Adams!") == "douglas adams"
    assert norm("St John's College") == "st johns college"


def test_candidate_spans():
    spans = candidate_spans("Where was Douglas Adams born?")
    assert "Douglas Adams" in spans
    assert all(not s.startswith("Where") for s in spans)
    spans = candidate_spans("Is St John's College in England?")
    assert any("St John" in s for s in spans)


def test_format_value():
    assert format_value("time", "+1952-03-11T00:00:00Z|11", {}) == "1952-03-11"
    assert format_value("time", "+1511-00-00T00:00:00Z|9", {}) == "1511"
    assert format_value("quantity", "+900|1", {}) == "900 1"
    assert format_value("entity", "Q42", {"Q42": "Douglas Adams"}) == "Douglas Adams"
    assert format_value("entity", "Q404", {}) is None
    assert format_value("string", "writer", {}) == "writer"


def test_resolve_orders_by_sitelinks(corpus_dir: Path):
    store = Store.open(corpus_dir)
    hits = store.resolve("douglas adams")
    assert hits[0].qid == "Q42"
    assert hits[0].label == "Douglas Adams"
    assert store.resolve("nobody at all") == []


def test_evidence_by_qid_and_name(corpus_dir: Path):
    store = Store.open(corpus_dir)
    ev = store.evidence("Q42")
    assert ev.label == "Douglas Adams"
    assert "- place of birth: Cambridge" in ev.text
    assert "- date of birth: 1952-03-11" in ev.text
    by_name = store.evidence("Douglas Adams")
    assert by_name.text == ev.text


def test_object_drop_rule(corpus_dir: Path):
    """Triple whose object entity is outside installed buckets renders nothing."""
    store = Store.open(corpus_dir)
    ev = store.evidence("Q691283")
    assert "- student: Douglas Adams" in ev.text      # cross-bucket object resolves
    assert "part of" not in ev.text                   # Q9999999 not installed -> dropped
    assert "Q9999999" not in ev.text                  # never a bare Q-id


def test_level_truncation(corpus_dir: Path):
    """At L0 the bucket-2 subject has no evidence at all."""
    store = Store.open(corpus_dir, level=0)
    ev = store.evidence("Q691283")
    assert ev.text == ""
    assert "outside level 0" in ev.note
    # and bucket-0 entities still work
    assert store.evidence("Q42").text


def test_lookup_end_to_end(corpus_dir: Path):
    store = Store.open(corpus_dir)
    result = store.lookup("Where was Douglas Adams born?")
    assert result["entities"]
    block = result["entities"][0]
    assert block["qid"] == "Q42"
    assert "Cambridge" in block["text"]


def test_missing_corpus():
    with pytest.raises(FileNotFoundError):
        Store.open(Path("does/not/exist"))
