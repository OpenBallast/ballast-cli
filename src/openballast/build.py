"""Build a Ballast serving artifact from your own documents (BYO corpus).

Input: a directory of `.md` / `.txt` files (one document each; the title is
the first `# heading` or the filename stem) and/or `.parquet` files with a
`text` column plus optional `title` and `rank` columns.

Output: the same per-bucket SQLite layout `ballast pull` installs, so
`lookup`, `serve`, and `mcp` work against it unchanged:

    <BALLAST_HOME>/<name>/properties.sqlite
    <BALLAST_HOME>/<name>/bucket_0.sqlite ... bucket_<k>.sqlite

Documents become passage-backed entities: `entities` and `names` address a
document by its title, and a `passages` table holds its prose chunks. Chunks
are atomic — a paragraph run is merged greedily up to the chunk byte budget
and never truncated mid-chunk at serve time.

Rank semantics match the published corpus: `rank` in [0, 1] (1 = most
important) maps documents onto nested buckets 0..MAX_BUCKET so `--level`
keeps the highest-ranked slice. Documents without a rank all land in
bucket 0 and the level knob degrades to a no-op.
"""

from __future__ import annotations

import json
import re
import sqlite3
import unicodedata
from dataclasses import dataclass
from pathlib import Path

from .pull import MAX_BUCKET, ballast_home
from .store import norm

CHUNK_BYTES = 1200
_HEADING = re.compile(r"^#{1,6}\s+(.+?)\s*$", re.MULTILINE)
_MD_NOISE = re.compile(r"```.*?```|`[^`]*`|!\[[^\]]*\]\([^)]*\)", re.DOTALL)
_MD_LINK = re.compile(r"\[([^\]]+)\]\([^)]*\)")

# Bucket shares mirror the published corpus's log-spaced nesting: bucket 0
# holds the top slice of rank mass, each later bucket roughly doubles.
_BUCKET_SHARES = (0.005, 0.005, 0.01, 0.02, 0.04, 0.08, 0.16, 0.68)


@dataclass
class Doc:
    title: str
    text: str
    rank: float | None


def _read_text_doc(path: Path) -> Doc:
    raw = path.read_text(encoding="utf-8", errors="ignore")
    title = path.stem.replace("_", " ").replace("-", " ").strip()
    m = _HEADING.search(raw)
    if m:
        title = m.group(1).strip()
        raw = raw[: m.start()] + raw[m.end():]
    raw = _MD_NOISE.sub(" ", raw)
    raw = _MD_LINK.sub(r"\1", raw)
    return Doc(title=title, text=raw, rank=None)


def _read_parquet_docs(path: Path) -> list[Doc]:
    import pyarrow.parquet as pq

    table = pq.read_table(path)
    cols = set(table.column_names)
    if "text" not in cols:
        raise ValueError(f"{path.name}: parquet input needs a 'text' column")
    texts = table.column("text").to_pylist()
    titles = table.column("title").to_pylist() if "title" in cols else [None] * len(texts)
    ranks = table.column("rank").to_pylist() if "rank" in cols else [None] * len(texts)
    docs = []
    for i, (text, title, rank) in enumerate(zip(texts, titles, ranks)):
        if not text or not str(text).strip():
            continue
        docs.append(Doc(
            title=str(title).strip() if title else f"{path.stem} {i + 1}",
            text=str(text),
            rank=float(rank) if rank is not None else None,
        ))
    return docs


def scan_dir(src: Path) -> list[Doc]:
    docs: list[Doc] = []
    for path in sorted(src.rglob("*")):
        if not path.is_file():
            continue
        suffix = path.suffix.lower()
        if suffix in (".md", ".txt"):
            doc = _read_text_doc(path)
            if doc.text.strip():
                docs.append(doc)
        elif suffix == ".parquet":
            docs.extend(_read_parquet_docs(path))
    return docs


def chunk_text(text: str, budget: int = CHUNK_BYTES) -> list[str]:
    """Paragraph split + greedy merge up to `budget` bytes per chunk.

    A single paragraph longer than the budget is split on sentence-ish
    boundaries as a fallback so no chunk exceeds ~2x the budget.
    """
    text = unicodedata.normalize("NFC", text)
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks: list[str] = []
    cur = ""
    for p in paras:
        p = re.sub(r"\s+", " ", p)
        if len(p.encode()) > budget * 2:
            sentences = re.split(r"(?<=[.!?])\s+", p)
            merged = ""
            for s in sentences:
                if merged and len((merged + " " + s).encode()) > budget:
                    paras_extra = merged
                    if cur and len((cur + "\n" + paras_extra).encode()) > budget:
                        chunks.append(cur)
                        cur = ""
                    cur = (cur + "\n" + paras_extra).strip() if cur else paras_extra
                    chunks.append(cur)
                    cur = ""
                    merged = s
                else:
                    merged = (merged + " " + s).strip() if merged else s
            p = merged
            if not p:
                continue
        if cur and len((cur + "\n" + p).encode()) > budget:
            chunks.append(cur)
            cur = p
        else:
            cur = (cur + "\n" + p).strip() if cur else p
    if cur:
        chunks.append(cur)
    return chunks


def assign_buckets(docs: list[Doc]) -> list[int]:
    """Nested bucket per doc from rank (1 = best -> bucket 0). No ranks -> all 0."""
    if not any(d.rank is not None for d in docs):
        return [0] * len(docs)
    order = sorted(range(len(docs)),
                   key=lambda i: -(docs[i].rank if docs[i].rank is not None else 0.0))
    buckets = [0] * len(docs)
    n = len(docs)
    cum, edges = 0.0, []
    for share in _BUCKET_SHARES:
        cum += share
        edges.append(max(1, round(cum * n)))
    for pos, i in enumerate(order):
        b = next(k for k, edge in enumerate(edges) if pos < edge)
        buckets[i] = min(b, MAX_BUCKET)
    return buckets


def build(src: Path, name: str, out: Path | None = None,
          chunk_bytes: int = CHUNK_BYTES, quiet: bool = False) -> Path:
    docs = scan_dir(src)
    if not docs:
        raise FileNotFoundError(f"no .md/.txt/.parquet documents under {src}")
    dest = out or (ballast_home() / name)
    dest.mkdir(parents=True, exist_ok=True)

    buckets = assign_buckets(docs)
    used_buckets = sorted(set(buckets))

    props = sqlite3.connect(dest / "properties.sqlite")
    props.execute("CREATE TABLE IF NOT EXISTS properties (pid TEXT PRIMARY KEY, label TEXT)")
    props.commit()
    props.close()

    cons: dict[int, sqlite3.Connection] = {}
    for b in used_buckets:
        con = sqlite3.connect(dest / f"bucket_{b}.sqlite")
        con.executescript(
            "DROP TABLE IF EXISTS entities; DROP TABLE IF EXISTS names;"
            "DROP TABLE IF EXISTS triples; DROP TABLE IF EXISTS passages;"
            "CREATE TABLE entities (qid TEXT PRIMARY KEY, label TEXT, bucket INTEGER);"
            "CREATE TABLE names (lname TEXT, qid TEXT, sitelinks INTEGER);"
            "CREATE TABLE triples (qid TEXT, pid TEXT, value_type TEXT, value TEXT, bucket INTEGER);"
            "CREATE TABLE passages (qid TEXT, chunk_idx INTEGER, text TEXT, bucket INTEGER);"
            "CREATE INDEX idx_names ON names (lname);"
            "CREATE INDEX idx_passages ON passages (qid, chunk_idx);"
        )
        cons[b] = con

    n_chunks = 0
    for i, (doc, b) in enumerate(zip(docs, buckets)):
        qid = f"D{i + 1}"
        con = cons[b]
        con.execute("INSERT INTO entities VALUES (?, ?, ?)", (qid, doc.title, b))
        lname = norm(doc.title)
        if lname:
            con.execute("INSERT INTO names VALUES (?, ?, ?)", (lname, qid, 1))
        for idx, chunk in enumerate(chunk_text(doc.text, chunk_bytes)):
            con.execute("INSERT INTO passages VALUES (?, ?, ?, ?)", (qid, idx, chunk, b))
            n_chunks += 1

    for b, con in cons.items():
        con.commit()
        con.close()

    manifest = {
        "kind": "byo", "name": name, "source": str(src),
        "documents": len(docs), "chunks": n_chunks,
        "buckets": used_buckets, "chunk_bytes": chunk_bytes,
        "ranked": any(d.rank is not None for d in docs),
        "files": {p.name: p.stat().st_size for p in dest.glob("*.sqlite")},
    }
    (dest / "installed.json").write_text(json.dumps(manifest, indent=2))
    if not quiet:
        sizes = sum(manifest["files"].values()) / 1e6
        print(f"built '{name}': {len(docs)} documents, {n_chunks} chunks, "
              f"{len(used_buckets)} bucket(s), {sizes:.1f} MB -> {dest}")
    return dest
