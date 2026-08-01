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


def test_lowercase_fallback_mining_default(corpus_dir: Path):
    """All-lowercase question mines nothing capitalized; n-gram fallback links
    by default (wrong links measured ~neutral, so recall pays)."""
    store = Store.open(corpus_dir)
    pairs = store.link("where was douglas adams born")
    assert pairs and pairs[0][1].qid == "Q42"
    assert store.link("where was douglas adams born", fallback=False) == []


def test_capitalized_path_still_works(corpus_dir: Path):
    store = Store.open(corpus_dir)
    pairs = store.link("Where was Douglas Adams born?")
    assert pairs[0][1].qid == "Q42"


def test_fallback_no_false_links(corpus_dir: Path):
    store = Store.open(corpus_dir, retriever="r2")
    assert store.link("why is water wet") == []


def test_r3_fts_partial_name(corpus_dir: Path):
    """Exact equality fails on possessives/partial names; FTS recovers."""
    r2 = Store.open(corpus_dir, retriever="r2")
    assert r2.resolve("Douglas Adams's") == []
    r3 = Store.open(corpus_dir, retriever="r3")
    hits = r3.resolve("Douglas Adams's")
    assert hits and hits[0].qid == "Q42"
    # exact matches are unaffected by the sidecar
    assert r3.resolve("Douglas Adams")[0].qid == "Q42"
