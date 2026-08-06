"""Competence profiling: estimator shape, AUC gate, budget advisor."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from openballast.profile import _auc, advise_level, run_profile, subject_features
from openballast.store import Store


class BucketOracle:
    """Knows bucket-0 subjects, blanks on bucket-2 — a clean competence split."""

    def __init__(self, known_golds: dict[str, str]):
        self.model = "oracle:1b"
        self.base_url = "http://fake"
        self._known = known_golds

    def complete(self, prompt: str, max_tokens: int = 24) -> str:
        for needle, gold in self._known.items():
            if needle in prompt:
                return gold
        return "unknown"


def _probe_rows() -> list[dict]:
    # Subjects span both buckets on BOTH sides of the deterministic subject-
    # hash split (fit: Q42, Q350, Q308; eval: Q21, Q691283, Q80503, Q15862).
    head = [("Q42", "Douglas Adams"), ("Q350", "Cambridge"), ("Q21", "England")]
    tail = [("Q691283", "St John's College"), ("Q308", "Mercury planet"),
            ("Q80503", "Mercury singer"), ("Q15862", "Queen band")]
    rows = []
    for qid, name in head:
        for i in range(8):
            rows.append({"question_id": f"h{qid}{i}", "question": f"{name} fact {i}?",
                         "gold": "yes", "gold_aliases": [], "subj_qid": qid,
                         "prop": "P19"})
    for qid, name in tail:
        for i in range(8):
            rows.append({"question_id": f"t{qid}{i}", "question": f"{name} fact {i}?",
                         "gold": "yes", "gold_aliases": [], "subj_qid": qid,
                         "prop": "P17"})
    return rows


@pytest.fixture()
def probes_path(tmp_path: Path) -> Path:
    import pyarrow as pa
    import pyarrow.parquet as pq

    path = tmp_path / "probes.parquet"
    pq.write_table(pa.Table.from_pylist(_probe_rows()), path)
    return path


def test_subject_features(corpus_dir: Path):
    store = Store.open(corpus_dir)
    feats = subject_features(store, ["Q42", "Q691283", "Q9999999"])
    assert feats["Q42"]["bucket"] == 0 and feats["Q42"]["sitelinks"] == 120
    assert feats["Q691283"]["bucket"] == 2
    assert feats["Q42"]["n_claims"] == 3
    assert "Q9999999" not in feats  # outside the installed corpus


def test_run_profile_separates_buckets(corpus_dir: Path, probes_path: Path, tmp_path: Path):
    store = Store.open(corpus_dir)
    up = BucketOracle({"Douglas Adams": "yes", "Cambridge": "yes", "England": "yes"})
    out = tmp_path / "m.gcp.json"
    prof = run_profile(up, store, str(probes_path), limit=0, out=out, quiet=True)

    assert prof["kind"] == "gcp"
    marg = prof["bucket_marginals"]
    assert marg["0"]["p"] == 1.0 and marg["2"]["p"] == 0.0
    # perfect bucket separation -> AUC well above the gate
    assert prof["reliability_auc"] >= prof["auc_gate"]
    assert prof["reliable"] is True
    assert json.loads(out.read_text())["model"] == "oracle:1b"


def test_auc_ties_and_degenerate():
    assert _auc([0.5, 0.5, 0.5], [True, False, True]) == pytest.approx(0.5)
    assert _auc([0.9, 0.1], [True, True]) == pytest.approx(0.5)  # one class
    assert _auc([0.9, 0.1], [True, False]) == pytest.approx(1.0)


def test_advise_level_budget_and_competence(corpus_dir: Path):
    sizes = {b: (corpus_dir / f"bucket_{b}.sqlite").stat().st_size for b in (0, 2)}
    # model already competent in bucket 0 (p >= .9) -> bytes go deeper
    prof = {"bucket_marginals": {"0": {"n": 12, "p": 0.95}, "2": {"n": 12, "p": 0.1}}}
    assert advise_level(prof, corpus_dir, sizes[0] + sizes[2], quiet=True) == 2
    # tiny budget: nothing past bucket 0 fits -> stays at 0
    assert advise_level(prof, corpus_dir, 1, quiet=True) == 0
