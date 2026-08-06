from openballast.probes import grade, load_probes, normalize


def test_normalize():
    assert normalize("The Old Man & the Sea!") == "old man sea"
    assert normalize("  Ernest   Hemingway ") == "ernest hemingway"


def test_grade_exact_and_containment():
    em, correct = grade("Paris", "Paris", [])
    assert em and correct
    em, correct = grade("the city of Paris, France", "Paris", [])
    assert not em and correct
    em, correct = grade("journalism", "journalist", ["journo"])
    assert not correct
    em, correct = grade("", "Paris", [])
    assert not correct


def test_load_probes_local_parquet(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq

    rows = [{"question_id": f"q{i}", "question": "?", "gold": "x",
             "gold_aliases": ["x"], "subj_qid": f"Q{i}", "prop": "p"}
            for i in range(20)]
    p = tmp_path / "probes.parquet"
    pq.write_table(pa.Table.from_pylist(rows), p)
    out = load_probes(str(p), limit=5)
    assert len(out) == 5
    # deterministic sample
    assert [r["question_id"] for r in out] == \
        [r["question_id"] for r in load_probes(str(p), limit=5)]
