"""Tiny fixture corpus in the per-bucket serving layout.

Bucket 0: Douglas Adams (Q42), Cambridge (Q350), England (Q21).
Bucket 2: St John's College (Q691283) — evidence includes an object (Q42) from a
          lower bucket, and one triple whose object (Q9999999, bucket 7) is not
          installed, exercising the object-drop rule.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

SCHEMA = """
CREATE TABLE entities (
    qid TEXT PRIMARY KEY, label TEXT, bucket INTEGER, sitelinks INTEGER
) WITHOUT ROWID;
CREATE TABLE names (
    lname TEXT, qid TEXT, sitelinks INTEGER, PRIMARY KEY (lname, qid)
) WITHOUT ROWID;
CREATE TABLE triples (
    qid TEXT, pid TEXT, value_type TEXT, value TEXT, bucket INTEGER,
    PRIMARY KEY (qid, pid, value_type, value, bucket)
) WITHOUT ROWID;
"""

BUCKET0 = {
    "entities": [
        ("Q42", "Douglas Adams", 0, 120),
        ("Q350", "Cambridge", 0, 200),
        ("Q21", "England", 0, 300),
    ],
    "names": [
        ("douglas adams", "Q42", 120),
        ("douglas noel adams", "Q42", 120),
        ("cambridge", "Q350", 200),
        ("england", "Q21", 300),
    ],
    "triples": [
        ("Q42", "P19", "entity", "Q350", 0),
        ("Q42", "P569", "time", "+1952-03-11T00:00:00Z|11", 0),
        ("Q42", "P106", "string", "writer", 0),
        ("Q350", "P17", "entity", "Q21", 0),
    ],
}

BUCKET2 = {
    "entities": [("Q691283", "St John's College", 2, 40)],
    "names": [("st johns college", "Q691283", 40)],
    "triples": [
        ("Q691283", "P17", "entity", "Q21", 2),
        ("Q691283", "P802", "entity", "Q42", 2),
        # object entity lives in bucket 7 — never installed in the fixture
        ("Q691283", "P361", "entity", "Q9999999", 2),
        ("Q691283", "P571", "time", "+1511-00-00T00:00:00Z|9", 2),
        ("Q691283", "P2124", "quantity", "+900|1", 2),
    ],
}


def write_bucket(path: Path, data: dict) -> None:
    db = sqlite3.connect(path)
    db.executescript(SCHEMA)
    db.executemany("INSERT INTO entities VALUES (?,?,?,?)", data["entities"])
    db.executemany("INSERT INTO names VALUES (?,?,?)", data["names"])
    db.executemany("INSERT INTO triples VALUES (?,?,?,?,?)", data["triples"])
    db.commit()
    db.close()


@pytest.fixture(scope="session")
def corpus_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    d = tmp_path_factory.mktemp("t0")
    props = sqlite3.connect(d / "properties.sqlite")
    props.executescript("CREATE TABLE properties (pid TEXT PRIMARY KEY, label TEXT);")
    props.executemany(
        "INSERT INTO properties VALUES (?,?)",
        [
            ("P19", "place of birth"), ("P569", "date of birth"),
            ("P106", "occupation"), ("P17", "country"), ("P802", "student"),
            ("P361", "part of"), ("P571", "inception"), ("P2124", "member count"),
        ],
    )
    props.commit()
    props.close()
    write_bucket(d / "bucket_0.sqlite", BUCKET0)
    write_bucket(d / "bucket_2.sqlite", BUCKET2)
    return d
