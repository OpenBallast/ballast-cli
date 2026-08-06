"""Three-arm instrument: arm mechanics, delivery arithmetic, checkpointing."""

from __future__ import annotations

import json
from pathlib import Path

import pyarrow.parquet as pq
import pytest

from openballast.instrument import run_arm, run_eval
from openballast.store import Store


class FakeUpstream:
    """Knows Douglas Adams facts only when the prompt carries evidence."""

    def __init__(self, model: str = "fake:1b"):
        self.model = model
        self.calls = 0

    def complete(self, prompt: str, max_tokens: int = 24) -> str:
        self.calls += 1
        question = prompt.rsplit("Q:", 1)[1]
        evidence = "Douglas Adams" in prompt.rsplit("\n\n", 2)[-2] if "Facts about" in prompt else False
        if "capital of England" in question:
            return "London"  # parametric: always known
        if "Douglas Adams born" in question:
            # knowledge-limited: right only when grounded with his facts
            return "Cambridge" if "Cambridge" in prompt else "Oxford"
        return "no idea"


@pytest.fixture()
def probes_path(tmp_path: Path) -> Path:
    import pyarrow as pa

    rows = [
        {"question_id": "p1", "question": "What is the capital of England?",
         "gold": "London", "gold_aliases": [], "subj_qid": "Q21", "prop": "P36"},
        {"question_id": "p2", "question": "Where was Douglas Adams born?",
         "gold": "Cambridge", "gold_aliases": ["cambridge, england"],
         "subj_qid": "Q42", "prop": "P19"},
        {"question_id": "p3", "question": "Who was the first ruler of Q9999999?",
         "gold": "Nobody", "gold_aliases": [], "subj_qid": "Q9999999", "prop": "P35"},
    ]
    path = tmp_path / "probes.parquet"
    pq.write_table(pa.Table.from_pylist(rows), path)
    return path


def test_run_eval_arms_and_delivery(corpus_dir: Path, probes_path: Path, tmp_path: Path):
    store = Store.open(corpus_dir)
    up = FakeUpstream()
    outdir = tmp_path / "eval"
    summary = run_eval(up, store, str(probes_path), limit=0, outdir=outdir, quiet=True)

    tol = 2e-4  # summary values are rounded to 4 decimals
    assert summary["n"] == 3
    # U: p1 right (parametric), p2 wrong (Oxford), p3 wrong
    assert summary["accuracy"]["ungrounded"] == pytest.approx(1 / 3, abs=tol)
    # S: p2 gets oracle Adams evidence -> right; p3 subject not installed -> wrong
    assert summary["accuracy"]["saturated"] == pytest.approx(2 / 3, abs=tol)
    band = summary["knowledge_limited_band"]
    assert band == pytest.approx(1 / 3, abs=tol)
    # delivery = (R - U) / band, whatever the linker realized
    r = summary["accuracy"]["realized"]
    assert summary["delivery_ratio"] == pytest.approx((r - 1 / 3) / band, abs=1e-3)
    # only p2's subject (Q42) yields saturated evidence: Q21 has no triples
    # as subject, Q9999999 is outside the corpus
    assert summary["coverage"] == pytest.approx(1 / 3, abs=tol)
    assert summary["covered"]["n"] == 1
    assert (outdir / "summary.json").exists()
    assert json.loads((outdir / "summary.json").read_text()) == summary


def test_arm_checkpoint_resume(corpus_dir: Path, probes_path: Path, tmp_path: Path):
    from openballast.probes import load_probes

    store = Store.open(corpus_dir)
    probes = load_probes(str(probes_path))
    outdir = tmp_path / "arms"
    outdir.mkdir()

    up = FakeUpstream()
    rows1 = run_arm(up, store, probes, "ungrounded", outdir, quiet=True)
    calls_first = up.calls
    assert calls_first == len(probes)

    rows2 = run_arm(up, store, probes, "ungrounded", outdir, quiet=True)
    assert up.calls == calls_first  # banked arm never re-queries
    assert rows2 == rows1


def test_evidence_has_gold_recorded(corpus_dir: Path, probes_path: Path, tmp_path: Path):
    from openballast.probes import load_probes

    store = Store.open(corpus_dir)
    probes = [p for p in load_probes(str(probes_path)) if p["question_id"] == "p2"]
    rows = run_arm(FakeUpstream(), store, probes, "saturated", tmp_path, quiet=True)
    assert rows[0]["evidence_bytes"] > 0
    assert rows[0]["evidence_has_gold"] is True
