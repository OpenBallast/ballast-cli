from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from openballast.build import assign_buckets, build, chunk_text, Doc, scan_dir
from openballast.store import Store


@pytest.fixture()
def docs_dir(tmp_path: Path) -> Path:
    src = tmp_path / "docs"
    src.mkdir()
    (src / "espresso_machine.md").write_text(
        "# Espresso Machine Maintenance\n\n"
        "The group head should be backflushed weekly with a blind filter.\n\n"
        "Descaling uses citric acid at 20 grams per liter of water. "
        "Never use vinegar on aluminum boilers.\n",
        encoding="utf-8",
    )
    (src / "grinder.txt").write_text(
        "Burr grinders should be recalibrated after every 20 kilograms of "
        "beans. The zero point drifts as burrs wear.\n",
        encoding="utf-8",
    )
    pq.write_table(pa.Table.from_pylist([
        {"title": "Water Filter", "text": "Replace the water filter every "
         "two months or 200 liters, whichever comes first.", "rank": 0.9},
        {"title": "Steam Wand", "text": "Purge the steam wand before and "
         "after every use to keep milk out of the boiler.", "rank": 0.2},
    ]), src / "extra.parquet")
    return src


def test_scan_dir_reads_all_formats(docs_dir: Path):
    docs = scan_dir(docs_dir)
    titles = {d.title for d in docs}
    assert "Espresso Machine Maintenance" in titles  # heading beats filename
    assert "grinder" in titles
    assert "Water Filter" in titles and "Steam Wand" in titles


def test_chunk_text_budget_and_atomicity():
    text = "\n\n".join(f"Paragraph {i} " + "word " * 40 for i in range(10))
    chunks = chunk_text(text, budget=400)
    assert len(chunks) > 1
    assert all(len(c.encode()) <= 800 for c in chunks)
    # nothing lost
    assert sum(c.count("Paragraph") for c in chunks) == 10


def test_assign_buckets_uniform_without_ranks():
    docs = [Doc("a", "x", None), Doc("b", "y", None)]
    assert assign_buckets(docs) == [0, 0]


def test_assign_buckets_ranked_orders_head_first():
    docs = [Doc(f"d{i}", "x", rank=1.0 - i / 100) for i in range(100)]
    buckets = assign_buckets(docs)
    assert buckets[0] == 0            # best rank -> head bucket
    assert buckets[99] >= buckets[0]  # worst rank never above best
    assert max(buckets) > 0           # spread across levels


def test_build_end_to_end_lookup(docs_dir: Path, tmp_path: Path):
    out = tmp_path / "corpus"
    build(docs_dir, name="shop", out=out, quiet=True)
    store = Store.open(out)
    ev = store.evidence("Espresso Machine Maintenance")
    assert "citric acid" in ev.text
    result = store.lookup("How do I descale the Espresso Machine Maintenance boiler?")
    assert result["entities"]
    assert any("citric acid" in b["text"] for b in result["entities"])
    hits = store.resolve("water filter")
    assert hits and store.evidence(hits[0].qid).text
