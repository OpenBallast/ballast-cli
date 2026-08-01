"""Read-only access to the Ballast serving artifact (per-bucket SQLite).

Layout (downloaded by `ballast pull` from the HF dataset's serving/sqlite tree):

    ~/.ballast/t0/properties.sqlite
    ~/.ballast/t0/bucket_0.sqlite ... bucket_7.sqlite

Each bucket database holds that rank bucket's `entities`, `names`, and `triples`
tables. A level Lk attaches properties + buckets 0..k — the corpus-quantization
knob. Truncation semantics match the parquet artifact and the live demo endpoint:
a subject outside the level has no evidence, and a triple whose object entity
falls outside the level is dropped, never rendered as a bare Q-id.

The name normalization and evidence rendering mirror the reference server
(mcp.openballast.org), so local and hosted lookups return identical blocks.
"""

from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path

MAX_BUCKET = 7

_NORM_STRIP = re.compile(r"[^a-z0-9 ]")
_NORM_SPACES = re.compile(r" +")

_SPAN = re.compile(
    r"[A-Z][\w'’.\-]*"
    r"(?:\s+(?:of|the|de|von|van|and|&|for|in)\s+[A-Z][\w'’.\-]*"
    r"|\s+[A-Z0-9][\w'’.\-]*)*"
)
_SPAN_TRIM = re.compile(r"^[\s.,?!'’\"]+|[\s.,?!'’\"]+$")
_STOP = {
    "The", "What", "Who", "Where", "When", "Which", "How", "Why", "In", "On",
    "At", "A", "An", "Is", "Was", "Are", "Did", "Does", "During",
}


_CONTENT_STOP = frozenset(
    """a an the of in on at to for and or nor is was are were be been being who
    whom what when where which how why did does do done can could would should
    shall will may might must it its this that these those with from by as not
    no s t d ll re ve o also into over under about after before between during
    than then there their his her they them he she you your i we our us""".split()
)


def norm(s: str) -> str:
    """Same normalization as the corpus name index (and the research linker)."""
    return _NORM_SPACES.sub(" ", _NORM_STRIP.sub("", s.lower())).strip()


def content_tokens(text: str) -> set[str]:
    """Normalized tokens minus stopwords — the disambiguation vocabulary."""
    return {t for t in norm(text).split() if t not in _CONTENT_STOP and len(t) > 1}


def candidate_spans(question: str) -> list[str]:
    """Mine capitalized entity mentions from a question, longest first."""
    spans: set[str] = set()
    for m in _SPAN.finditer(question):
        parts = _SPAN_TRIM.sub("", m.group(0)).split()
        while parts and parts[0] in _STOP:
            parts = parts[1:]
        if parts:
            spans.add(" ".join(parts))
    return sorted(spans, key=len, reverse=True)


def format_value(value_type: str, value: str, labels: dict[str, str]) -> str | None:
    """Render one triple object; entity objects without an in-level label are dropped."""
    if value_type == "entity":
        return labels.get(value)
    if value_type == "time":
        date = value.split("|")[0].lstrip("+").split("T")[0]
        if date.startswith("-"):
            return date
        parts = date.split("-")
        y = parts[0] if parts else ""
        m = parts[1] if len(parts) > 1 else ""
        d = parts[2] if len(parts) > 2 else ""
        if not m or m == "00":
            return y
        if not d or d == "00":
            return f"{y}-{m}"
        return f"{y}-{m}-{d}"
    if value_type == "quantity":
        return value.replace("|", " ").lstrip("+")
    return value


@dataclass
class Hit:
    qid: str
    label: str
    bucket: int
    sitelinks: int


@dataclass
class Evidence:
    qid: str | None
    level: int
    text: str = ""
    label: str | None = None
    note: str | None = None


@dataclass
class Store:
    """One read-only connection with properties + buckets 0..level attached."""

    data_dir: Path
    level: int
    _con: sqlite3.Connection = field(repr=False, default=None)  # type: ignore[assignment]
    _buckets: list[int] = field(default_factory=list)
    _fts_buckets: list[int] = field(default_factory=list)
    _fts_con: sqlite3.Connection | None = field(repr=False, default=None)

    @staticmethod
    def installed_buckets(data_dir: Path) -> list[int]:
        return sorted(
            b for b in range(MAX_BUCKET + 1) if (data_dir / f"bucket_{b}.sqlite").exists()
        )

    @classmethod
    def open(cls, data_dir: Path, level: int | None = None,
             retriever: str = "auto") -> "Store":
        data_dir = Path(data_dir)
        props = data_dir / "properties.sqlite"
        if not props.exists():
            raise FileNotFoundError(
                f"no corpus at {data_dir} — run `ballast pull` first"
            )
        available = cls.installed_buckets(data_dir)
        if not available:
            raise FileNotFoundError(f"no bucket databases at {data_dir}")
        if level is None:
            level = max(available)
        buckets = [b for b in available if b <= level]
        if not buckets:
            raise ValueError(f"level {level} below smallest installed bucket {available[0]}")

        con = sqlite3.connect(":memory:", check_same_thread=False)
        con.execute("ATTACH DATABASE ? AS pr", (props.as_posix(),))
        for b in buckets:
            con.execute(
                "ATTACH DATABASE ? AS b" + str(b),
                ((data_dir / f"bucket_{b}.sqlite").as_posix(),),
            )
        # R3 retriever rung: optional FTS5 sidecars (fts_{b}.sqlite) widen
        # candidate generation beyond exact-normalized-name equality. They get
        # their own connection: the main one already uses 9 of sqlite's 10
        # ATTACH slots at L7.
        fts_buckets: list[int] = []
        fts_con: sqlite3.Connection | None = None
        if retriever in ("auto", "r3"):
            present = [b for b in buckets if (data_dir / f"fts_{b}.sqlite").exists()]
            if present:
                fts_con = sqlite3.connect(":memory:", check_same_thread=False)
                for b in present:
                    fts_con.execute(
                        "ATTACH DATABASE ? AS f" + str(b),
                        ((data_dir / f"fts_{b}.sqlite").as_posix(),),
                    )
                fts_con.execute("PRAGMA query_only = ON")
                fts_buckets = present
            if retriever == "r3" and not fts_buckets:
                raise FileNotFoundError(f"retriever=r3 but no fts_*.sqlite at {data_dir}")
        con.execute("PRAGMA query_only = ON")
        store = cls(data_dir=data_dir, level=level)
        store._con = con
        store._buckets = buckets
        store._fts_buckets = fts_buckets
        store._fts_con = fts_con
        return store

    def close(self) -> None:
        if self._con is not None:
            self._con.close()
        if self._fts_con is not None:
            self._fts_con.close()

    def _union(self, select: str) -> str:
        return " UNION ALL ".join(select.format(b=f"b{b}") for b in self._buckets)

    # -- tools ------------------------------------------------------------

    def resolve(self, name: str, limit: int = 5, allow_fts: bool = True) -> list[Hit]:
        hits, _ = self._resolve_traced(name, limit, allow_fts)
        return hits

    def _resolve_traced(self, name: str, limit: int = 5,
                        allow_fts: bool = True) -> tuple[list[Hit], bool]:
        """Resolve, reporting whether the (fuzzier) FTS path produced the rows."""
        lname = norm(name)
        if not lname:
            return [], False
        sql = (
            "SELECT qid, sitelinks FROM ("
            + self._union("SELECT qid, sitelinks FROM {b}.names WHERE lname = ?")
            + ") ORDER BY sitelinks DESC LIMIT ?"
        )
        rows = self._con.execute(sql, [lname] * len(self._buckets) + [limit]).fetchall()
        fts_used = False
        if not rows and allow_fts and self._fts_buckets:
            rows = self._resolve_fts_rows(lname, limit)
            fts_used = bool(rows)
        hits = []
        for qid, sitelinks in rows:
            ent = self._entity(qid)
            if ent is not None:
                hits.append(Hit(qid=qid, label=ent[0], bucket=ent[1], sitelinks=sitelinks))
        return hits, fts_used

    def _resolve_fts_rows(self, lname: str, limit: int) -> list[tuple[str, int]]:
        """R3: bm25-ranked partial-name match when exact equality finds nothing
        ("barack obamas" -> "barack obama"). OR-query so extra/missing words
        don't zero the match; bm25 rank favors short, mostly-covered names."""
        tokens = [t for t in lname.split() if t]
        if not tokens:
            return []
        match = " OR ".join(f'"{t}"' for t in tokens)
        scored: list[tuple[float, str, int]] = []
        for b in self._fts_buckets:
            rows = self._fts_con.execute(
                f"SELECT rank, qid, sitelinks FROM f{b}.names_fts "
                "WHERE names_fts MATCH ? ORDER BY rank LIMIT ?",
                (match, limit),
            ).fetchall()
            scored.extend(rows)
        scored.sort(key=lambda r: r[0])  # bm25: lower = better
        out: list[tuple[str, int]] = []
        seen: set[str] = set()
        for _, qid, sitelinks in scored:
            if qid in seen:
                continue
            seen.add(qid)
            out.append((qid, sitelinks))
            if len(out) >= limit:
                break
        return out

    def _entity(self, qid: str) -> tuple[str, int] | None:
        sql = self._union("SELECT label, bucket FROM {b}.entities WHERE qid = ?")
        row = self._con.execute(sql, [qid] * len(self._buckets)).fetchone()
        return (row[0], row[1]) if row else None

    def _entity_labels(self, qids: list[str]) -> dict[str, str]:
        labels: dict[str, str] = {}
        for i in range(0, len(qids), 90):
            chunk = qids[i : i + 90]
            ph = ",".join("?" * len(chunk))
            sql = self._union(
                "SELECT qid, label FROM {b}.entities WHERE qid IN (" + ph + ")"
            )
            for qid, label in self._con.execute(sql, chunk * len(self._buckets)):
                labels[qid] = label
        return labels

    def _property_labels(self, pids: list[str]) -> dict[str, str]:
        labels: dict[str, str] = {}
        for i in range(0, len(pids), 90):
            chunk = pids[i : i + 90]
            ph = ",".join("?" * len(chunk))
            sql = "SELECT pid, label FROM pr.properties WHERE pid IN (" + ph + ")"
            for pid, label in self._con.execute(sql, chunk):
                labels[pid] = label
        return labels

    def evidence(self, id: str, max_triples: int = 32) -> Evidence:
        qid = id
        if not re.fullmatch(r"Q\d+", id):
            hits = self.resolve(id, 1)
            if not hits:
                return Evidence(qid=None, level=self.level, note=f"no entity named '{id}'")
            qid = hits[0].qid

        ent = self._entity(qid)
        if ent is None:
            return Evidence(
                qid=qid, level=self.level,
                note=f"entity outside level {self.level} (or unknown)",
            )
        label, _bucket = ent

        sql = (
            "SELECT pid, value_type, value FROM ("
            + self._union(
                "SELECT pid, value_type, value FROM {b}.triples WHERE qid = ? AND bucket <= ?"
            )
            + ") ORDER BY CAST(substr(pid, 2) AS INTEGER)"
        )
        triples = self._con.execute(sql, [qid, self.level] * len(self._buckets)).fetchall()
        if not triples:
            return Evidence(qid=qid, level=self.level, note="no triples at this level")

        object_qids = sorted({v for _, vt, v in triples if vt == "entity"})
        labels = self._entity_labels(object_qids)
        prop_labels = self._property_labels(sorted({p for p, _, _ in triples}))

        lines: list[str] = []
        for pid, value_type, value in triples:
            if len(lines) >= max_triples:
                break
            prop = prop_labels.get(pid)
            if not prop:
                continue
            rendered = format_value(value_type, value, labels)
            if not rendered:
                continue
            lines.append(f"- {prop}: {rendered}")
        if not lines:
            return Evidence(qid=qid, level=self.level, note="no renderable triples at this level")
        return Evidence(
            qid=qid, level=self.level, label=label,
            text=f"Facts about {label}:\n" + "\n".join(lines),
        )

    def resolve_ctx(self, name: str, question: str, k: int = 8) -> Hit | None:
        """Resolve with context disambiguation (see _resolve_ctx_scored)."""
        scored = self._resolve_ctx_scored(name, question, k)
        return scored[0] if scored else None

    def _resolve_ctx_scored(self, name: str, question: str, k: int = 8
                            ) -> tuple[Hit, int, int, bool, bool] | None:
        """Context disambiguation with provenance: returns (hit, overlap,
        n_candidates, dominant, fts_used). Among the top-k name candidates,
        prefer the one whose evidence shares the most content words with the
        question; blended with log-sitelinks so one spurious shared token can't
        dethrone a strongly notable candidate. `dominant` = the top candidate
        outranks the runner-up >=5x on sitelinks (the "obviously the famous
        one" case, safe to attach without corroboration)."""
        hits, fts_used = self._resolve_traced(name, k)
        if not hits:
            return None
        if len(hits) == 1:
            return (hits[0], 0, 1, True, fts_used)
        import math

        dominant = hits[0].sitelinks >= 5 * max(1, hits[1].sitelinks)
        qtoks = content_tokens(question) - content_tokens(name)
        best: tuple[float, int, Hit] | None = None
        for h in hits:
            ev = self.evidence(h.qid, max_triples=24)
            overlap = len(qtoks & content_tokens(ev.text)) if ev.text else 0
            score = overlap + math.log10(1 + h.sitelinks)
            if best is None or score > best[0]:
                best = (score, overlap, h)
        # dominance only vouches for the sitelinks leader, not an overlap pick
        chosen_dominant = dominant and best[2].qid == hits[0].qid
        return (best[2], best[1], len(hits), chosen_dominant, fts_used)

    def _ngram_mentions(self, question: str, max_n: int = 5,
                        max_entities: int = 4) -> list[str]:
        """Lowercase fallback miner: when capitalized-span mining yields nothing
        (e.g. all-lowercase questions), find the longest non-overlapping word
        n-grams that exist in the names index."""
        toks = norm(question).split()
        grams: dict[str, tuple[int, int]] = {}  # gram -> (start, n)
        for n in range(min(max_n, len(toks)), 0, -1):
            for i in range(len(toks) - n + 1):
                window = toks[i : i + n]
                if all(t in _CONTENT_STOP for t in window):
                    continue
                if n == 1 and (len(window[0]) < 3 or window[0] in _CONTENT_STOP):
                    continue
                grams.setdefault(" ".join(window), (i, n))
        if not grams:
            return []
        existing: set[str] = set()
        names = sorted(grams)
        for i in range(0, len(names), 80):
            chunk = names[i : i + 80]
            ph = ",".join("?" * len(chunk))
            sql = self._union(
                "SELECT lname FROM {b}.names WHERE lname IN (" + ph + ")"
            )
            for (lname,) in self._con.execute(sql, chunk * len(self._buckets)):
                existing.add(lname)
        # greedy: longest grams first, no token overlap
        picked: list[str] = []
        used: set[int] = set()
        for g in sorted(existing, key=lambda g: (-grams[g][1], grams[g][0])):
            start, n = grams[g]
            span = set(range(start, start + n))
            if span & used:
                continue
            used |= span
            picked.append(g)
            if len(picked) >= max_entities:
                break
        return picked

    def link(self, question: str, max_entities: int = 4, ctx_k: int = 8,
             precision_gate: bool = False, fallback: bool = True) -> list[tuple[str, Hit]]:
        """Question -> [(mention, Hit)]: capitalized-span mining with blended
        context disambiguation, plus lowercase n-gram fallback when spans
        resolve to nothing. This configuration realizes ~63% of the oracle
        grounding gain end-to-end on the 50k-probe bench.

        Measured trade-offs (per-probe composition against banked model
        passes, wrong-link harm measured directly on GPU): attaching the WRONG
        entity's facts is ~neutral — the model ignores irrelevant evidence —
        so hit-rate is the objective and recall additions pay. (An earlier
        conclusion that wrong links were as harmful as right links were
        helpful traced to a padding-misaligned scorer; fixed and re-measured.)

        precision_gate (default off — measured to cost hits for no benefit):
        restricts fallback attaches to unambiguous / context-corroborated /
        sitelinks-dominant candidates. Kept for callers who want conservative
        linking in agent pipelines where evidence provenance matters."""

        def attach_ok(mention: str, overlap: int, n_cands: int, dominant: bool,
                      fts_used: bool, is_fallback: bool) -> bool:
            if not precision_gate:
                return True
            if fts_used and overlap < 1:
                return False
            if is_fallback and len(mention.split()) < 2:
                return False
            return n_cands == 1 or overlap >= 1 or dominant

        pairs: list[tuple[str, Hit]] = []
        seen: set[str] = set()
        for span in candidate_spans(question)[:max_entities]:
            scored = self._resolve_ctx_scored(span, question, ctx_k)
            if not scored:
                continue
            hit, overlap, n_cands, dominant, fts_used = scored
            if attach_ok(span, overlap, n_cands, dominant, fts_used, False) and hit.qid not in seen:
                seen.add(hit.qid)
                pairs.append((span, hit))
        if not pairs and fallback:
            for gram in self._ngram_mentions(question, max_entities=max_entities):
                scored = self._resolve_ctx_scored(gram, question, ctx_k)
                if not scored:
                    continue
                hit, overlap, n_cands, dominant, fts_used = scored
                if attach_ok(gram, overlap, n_cands, dominant, fts_used, True) and hit.qid not in seen:
                    seen.add(hit.qid)
                    pairs.append((gram, hit))
        return pairs

    def lookup(self, question: str, max_triples: int = 24, max_entities: int = 4) -> dict:
        blocks = []
        for mention, hit in self.link(question, max_entities=max_entities):
            ev = self.evidence(hit.qid, max_triples=max_triples)
            if ev.text:
                blocks.append({
                    "mention": mention, "qid": ev.qid, "label": ev.label,
                    "level": ev.level, "text": ev.text,
                })
        return {"level": self.level, "entities": blocks}
