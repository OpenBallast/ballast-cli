"""The three-arm instrument: measure what grounding actually delivers.

Every probe is asked three ways against the same model:

  U  ungrounded — no evidence; the parametric knowledge floor.
  R  realized   — evidence from the shipped linker (`Store.lookup`); what the
                  system delivers today.
  S  saturated  — evidence for the probe's TRUE subject entity (oracle
                  linking); the grounding ceiling for this corpus.

Derived: knowledge-limited band = S - U, and the number that matters,

    delivery_ratio = (R - U) / (S - U)

— the fraction of the reachable gap today's retrieval actually closes.
Coverage-conditional variants are reported when some subjects have no
evidence at the installed level. Arms checkpoint to parquet and are skipped
on rerun, so an interrupted pass resumes where it stopped.

`evidence_has_gold` is recorded per row as a descriptive statistic. It is
not, and must never become, an optimization target.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from .probes import answer_text, grade, load_probes
from .store import Store
from .upstream import Upstream

ARMS = ("ungrounded", "realized", "saturated")
MAX_EVIDENCE_BYTES = 2048

_SHOTS_PLAIN = (
    "Q: What is the capital of France?\nA: Paris\n\n"
    "Q: Who wrote The Old Man and the Sea?\nA: Ernest Hemingway\n\n"
)
_SHOTS_CTX = (
    "Facts about France:\n- capital: Paris\n\n"
    "Q: What is the capital of France?\nA: Paris\n\n"
)


def _pack(blocks: list[str], budget: int = MAX_EVIDENCE_BYTES) -> str:
    out: list[str] = []
    used = 0
    for b in blocks:
        cost = len(b.encode()) + (2 if out else 0)
        if used + cost > budget:
            continue  # atomic blocks: skip, never truncate
        out.append(b)
        used += cost
    return "\n\n".join(out)


def _evidence_for(store: Store, probe: dict, arm: str) -> str:
    if arm == "ungrounded":
        return ""
    if arm == "realized":
        result = store.lookup(probe["question"])
        return _pack([b["text"] for b in result["entities"]])
    ev = store.evidence(probe["subj_qid"])
    return _pack([ev.text]) if ev.text else ""


def _prompt(question: str, evidence: str) -> str:
    if evidence:
        return f"{_SHOTS_CTX}{evidence}\n\nQ: {question}\nA:"
    return f"{_SHOTS_PLAIN}Q: {question}\nA:"


def _arm_path(outdir: Path, model: str, arm: str) -> Path:
    safe = model.replace("/", "__").replace(":", "_")
    return outdir / f"arm__{safe}__{arm}.parquet"


def run_arm(upstream: Upstream, store: Store, probes: list[dict], arm: str,
            outdir: Path, quiet: bool = False) -> list[dict]:
    import pyarrow as pa
    import pyarrow.parquet as pq

    path = _arm_path(outdir, upstream.model, arm)
    if path.exists():
        if not quiet:
            print(f"  {arm}: already banked ({path.name})")
        return pq.read_table(path).to_pylist()

    rows = []
    t0 = time.time()
    for i, p in enumerate(probes, 1):
        evidence = _evidence_for(store, p, arm)
        pred = answer_text(upstream.complete(_prompt(p["question"], evidence)))
        em, correct = grade(pred, p["gold"], p["gold_aliases"])
        gold_in_ev = any(
            a and a.lower() in evidence.lower()
            for a in [p["gold"], *p["gold_aliases"]]
        ) if evidence else False
        rows.append({
            "question_id": p["question_id"], "arm": arm, "pred": pred[:120],
            "em": em, "correct": correct,
            "evidence_bytes": len(evidence.encode()),
            "evidence_has_gold": gold_in_ev,
        })
        if not quiet and i % 50 == 0:
            print(f"  {arm}: {i}/{len(probes)} "
                  f"({i / max(time.time() - t0, 1):.1f}/s)", flush=True)
    pq.write_table(pa.Table.from_pylist(rows), path)
    if not quiet:
        acc = sum(r["correct"] for r in rows) / len(rows)
        print(f"  {arm}: accuracy {acc:.4f} -> {path.name}")
    # the banked parquet is the source of truth for scoring — re-read it so
    # the summary can never disagree with what is on disk
    return pq.read_table(path).to_pylist()


def run_eval(upstream: Upstream, store: Store, evalset: str, limit: int,
             outdir: Path, quiet: bool = False) -> dict:
    outdir.mkdir(parents=True, exist_ok=True)
    probes = load_probes(evalset, limit=limit)
    if not quiet:
        print(f"three-arm instrument: {len(probes)} probes, model "
              f"{upstream.model}, corpus level L{store.level}")

    by_arm: dict[str, dict[str, dict]] = {}
    for arm in ARMS:
        rows = run_arm(upstream, store, probes, arm, outdir, quiet=quiet)
        by_arm[arm] = {r["question_id"]: r for r in rows}

    ids = [p["question_id"] for p in probes
           if all(p["question_id"] in by_arm[a] for a in ARMS)]
    n = len(ids)
    acc = {a: sum(by_arm[a][q]["correct"] for q in ids) / n for a in ARMS}
    band = acc["saturated"] - acc["ungrounded"]
    delivery = (acc["realized"] - acc["ungrounded"]) / band if band > 1e-9 else None

    covered = [q for q in ids if by_arm["saturated"][q]["evidence_bytes"] > 0]
    summary = {
        "model": upstream.model, "evalset": evalset, "n": n,
        "level": store.level,
        "accuracy": {a: round(acc[a], 4) for a in ARMS},
        "knowledge_limited_band": round(band, 4),
        "delivery_ratio": round(delivery, 4) if delivery is not None else None,
        "coverage": round(len(covered) / n, 4) if n else None,
    }
    if covered and len(covered) < n:
        cacc = {a: sum(by_arm[a][q]["correct"] for q in covered) / len(covered)
                for a in ARMS}
        cband = cacc["saturated"] - cacc["ungrounded"]
        summary["covered"] = {
            "n": len(covered),
            "accuracy": {a: round(cacc[a], 4) for a in ARMS},
            "delivery_ratio": round(
                (cacc["realized"] - cacc["ungrounded"]) / cband, 4
            ) if cband > 1e-9 else None,
        }
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2))
    if not quiet:
        print(f"\nU {acc['ungrounded']:.4f}  R {acc['realized']:.4f}  "
              f"S {acc['saturated']:.4f}")
        print(f"knowledge-limited band {band:+.4f}; delivery ratio "
              f"{delivery:.3f}" if delivery is not None else "band ~ 0")
        print(f"summary -> {outdir / 'summary.json'}")
    return summary
