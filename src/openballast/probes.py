"""Probe sets for `profile` and `eval` — downloaded from the public evalsets
dataset, or supplied as a local parquet with the same columns.

Required columns: question_id, question, gold, gold_aliases (list), subj_qid,
prop. Grading (`normalize`, `grade`) follows the SQuAD/NQ-open convention the
research harness uses: casefold, punctuation to spaces, articles dropped;
`correct` = any gold alias appears as a normalized substring of the answer.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

EVALSETS_REPO = "OpenBallast/ballast-evalsets"
EVALSETS = {"matrix": "matrix_probes.parquet", "halluc": "halluc_probes.parquet"}

_PUNCT = re.compile(r"[^\w\s]")
_ARTICLES = re.compile(r"\b(a|an|the)\b")
_SPACES = re.compile(r"\s+")


def normalize(s: str) -> str:
    s = _PUNCT.sub(" ", str(s).lower())
    s = _ARTICLES.sub(" ", s)
    return _SPACES.sub(" ", s).strip()


_THINK = re.compile(r"<think>.*?(?:</think>|$)", re.DOTALL)


def answer_text(raw: str) -> str:
    """Extract the answer from a model response: drop reasoning blocks, take
    the first non-empty line that remains."""
    s = _THINK.sub("", str(raw)).strip()
    for line in s.splitlines():
        line = line.strip()
        if line:
            return line
    return ""


def grade(pred: str, gold: str, aliases: list[str]) -> tuple[bool, bool]:
    """(exact_match, correct) — correct = any alias contained in the answer."""
    p = normalize(pred)
    golds = [normalize(g) for g in [gold, *aliases] if str(g).strip()]
    golds = [g for g in golds if g]
    if not p or not golds:
        return False, False
    em = any(p == g for g in golds)
    correct = em or any(g in p for g in golds)
    return em, correct


def fetch_evalset(name: str) -> Path:
    from huggingface_hub import hf_hub_download

    if name not in EVALSETS:
        raise ValueError(f"unknown evalset '{name}' (have: {', '.join(EVALSETS)})")
    return Path(hf_hub_download(
        repo_id=EVALSETS_REPO, repo_type="dataset", filename=EVALSETS[name],
    ))


def load_probes(source: str, limit: int | None = None,
                seed: int = 13) -> list[dict]:
    """Load probes from an evalset name or a local parquet path. `limit`
    takes a deterministic hash-ordered sample so reruns hit the same rows."""
    import pyarrow.parquet as pq

    path = Path(source)
    if not path.exists():
        path = fetch_evalset(source)
    table = pq.read_table(path)
    rows = table.to_pylist()
    seen: dict[str, int] = {}
    for r in rows:
        r["gold_aliases"] = [str(a) for a in (r.get("gold_aliases") or [])]
        # some evalsets carry duplicate question_ids (several relations under
        # one subject id) — uniquify so per-id joins across arms stay exact
        q = str(r["question_id"])
        n = seen.get(q, 0)
        seen[q] = n + 1
        if n:
            r["question_id"] = f"{q}#{n + 1}"
    if limit and limit < len(rows):
        rows.sort(key=lambda r: hashlib.sha1(
            f"{seed}:{r['question_id']}".encode()).hexdigest())
        rows = rows[:limit]
    return rows
