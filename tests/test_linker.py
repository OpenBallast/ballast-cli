from pathlib import Path

from openballast.store import Store, content_tokens


def test_content_tokens():
    toks = content_tokens("Who was the lead singer of the band Queen?")
    assert "singer" in toks and "queen" in toks and "band" in toks
    assert "the" not in toks and "of" not in toks


def test_ctx_disambiguation_prefers_context_match(corpus_dir: Path):
    """Two entities named Mercury; question about the band picks the tail one."""
    store = Store.open(corpus_dir)
    hit = store.resolve_ctx("Mercury", "Was Mercury a member of the rock band Queen?")
    assert hit.qid == "Q80503"
    hit = store.resolve_ctx("Mercury", "Is Mercury a planet of the solar system?")
    assert hit.qid == "Q308"
    # no context signal -> falls back to sitelinks order
    hit = store.resolve_ctx("Mercury", "Mercury?")
    assert hit.qid == "Q308"


def test_lowercase_fallback_mining(corpus_dir: Path):
    """All-lowercase question mines nothing capitalized; n-gram fallback links."""
    store = Store.open(corpus_dir)
    pairs = store.link("where was douglas adams born")
    assert pairs, "fallback should link the lowercase mention"
    assert pairs[0][1].qid == "Q42"
    # and the full lookup returns evidence through the same path
    result = store.lookup("where was douglas adams born")
    assert result["entities"] and result["entities"][0]["qid"] == "Q42"


def test_capitalized_path_still_works(corpus_dir: Path):
    store = Store.open(corpus_dir)
    pairs = store.link("Where was Douglas Adams born?")
    assert pairs[0][1].qid == "Q42"


def test_fallback_no_false_links(corpus_dir: Path):
    store = Store.open(corpus_dir)
    assert store.link("why is water wet") == []
