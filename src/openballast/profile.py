"""Competence profiling: measure WHERE a local model's parametric knowledge
runs out, per corpus region, and turn that into a level recommendation.

Method (the public form of the research estimator): probe the model
ungrounded on a sample of the evalset, join each probed subject's corpus
features (rank bucket, sitelinks, n_claims — features every corpus entity
has, never entity identity), and fit shrunk per-cell rates

    p(bucket, relation) = (n * rate + m * backoff) / (n + m)

with a bucket-marginal -> logistic(log1p sitelinks, log1p n_claims) backoff
chain. FIT/EVAL split is deterministic by subject hash so no subject's
relations straddle the boundary; the reliability AUC is reported from EVAL
only, against the 0.58 gate the research program uses: below it, the profile
is not reliable enough to steer corpus selection and the level advisor keeps
the generic ordering.

Output: `<out>.gcp.json` (grounding competence profile) + a per-bucket
competence table + a budget advisor mapping `--budget` bytes to the level
whose marginal accuracy per byte is still positive for THIS model.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from pathlib import Path

from .probes import answer_text, grade, load_probes
from .store import Store
from .upstream import Upstream

M_SHRINK = 25.0
AUC_GATE = 0.58


def _split(subj_qid: str, seed: int = 13) -> str:
    h = hashlib.sha1(f"{seed}:{subj_qid}".encode()).digest()
    return "fit" if h[0] % 2 == 0 else "eval"


def _logistic_fit(xs: list[list[float]], ys: list[float],
                  iters: int = 200, lr: float = 0.5) -> list[float]:
    w = [0.0, 0.0, 0.0]
    n = len(xs)
    for _ in range(iters):
        grad = [0.0, 0.0, 0.0]
        for x, y in zip(xs, ys):
            z = w[0] + w[1] * x[0] + w[2] * x[1]
            p = 1.0 / (1.0 + math.exp(-max(-30, min(30, z))))
            d = p - y
            grad[0] += d
            grad[1] += d * x[0]
            grad[2] += d * x[1]
        w = [wi - lr * g / n for wi, g in zip(w, grad)]
    return w


def _logistic_p(w: list[float], x: list[float]) -> float:
    z = w[0] + w[1] * x[0] + w[2] * x[1]
    return 1.0 / (1.0 + math.exp(-max(-30, min(30, z))))


def _auc(scores: list[float], labels: list[bool]) -> float:
    pairs = sorted(zip(scores, labels))
    ranks = {}
    i = 0
    while i < len(pairs):
        j = i
        while j < len(pairs) and pairs[j][0] == pairs[i][0]:
            j += 1
        r = (i + j + 1) / 2  # average rank, 1-based
        for k in range(i, j):
            ranks[k] = r
        i = j
    pos = [ranks[k] for k, (_, y) in enumerate(pairs) if y]
    n1, n0 = len(pos), len(pairs) - len(pos)
    if not n1 or not n0:
        return 0.5
    return (sum(pos) - n1 * (n1 + 1) / 2) / (n1 * n0)


def subject_features(store: Store, qids: list[str]) -> dict[str, dict]:
    """(bucket, sitelinks, n_claims) per subject, from the installed corpus."""
    feats: dict[str, dict] = {}
    for qid in qids:
        ent = store._entity(qid)
        if ent is None:
            continue
        _label, bucket = ent
        sql = (
            "SELECT sitelinks FROM ("
            + store._union("SELECT sitelinks FROM {b}.names WHERE qid = ?")
            + ") LIMIT 1"
        )
        row = store._con.execute(sql, [qid] * len(store._buckets)).fetchone()
        sitelinks = int(row[0]) if row else 0
        sql = store._union("SELECT COUNT(*) FROM {b}.triples WHERE qid = ?")
        n_claims = sum(
            r[0] for r in store._con.execute(sql, [qid] * len(store._buckets)).fetchall()
        )
        feats[qid] = {"bucket": bucket, "sitelinks": sitelinks, "n_claims": n_claims}
    return feats


def run_profile(upstream: Upstream, store: Store, evalset: str, limit: int,
                out: Path, quiet: bool = False) -> dict:
    probes = load_probes(evalset, limit=limit)
    feats = subject_features(store, sorted({p["subj_qid"] for p in probes}))
    probes = [p for p in probes if p["subj_qid"] in feats]
    if not quiet:
        print(f"probing {len(probes)} questions ungrounded "
              f"({len(feats)} subjects with corpus features)...")

    outcomes = []
    t0 = time.time()
    for i, p in enumerate(probes, 1):
        pred = answer_text(upstream.complete(f"Q: {p['question']}\nA:"))
        _em, correct = grade(pred, p["gold"], p["gold_aliases"])
        outcomes.append({**p, "correct": correct, "split": _split(p["subj_qid"])})
        if not quiet and i % 100 == 0:
            print(f"  {i}/{len(probes)} ({i / max(time.time() - t0, 1):.1f}/s)",
                  flush=True)

    fit = [o for o in outcomes if o["split"] == "fit"]
    ev = [o for o in outcomes if o["split"] == "eval"]

    w = _logistic_fit(
        [[math.log1p(feats[o["subj_qid"]]["sitelinks"]),
          math.log1p(feats[o["subj_qid"]]["n_claims"])] for o in fit],
        [float(o["correct"]) for o in fit],
    )
    bucket_marginals: dict[int, dict] = {}
    for o in fit:
        b = feats[o["subj_qid"]]["bucket"]
        m = bucket_marginals.setdefault(b, {"n": 0, "k": 0})
        m["n"] += 1
        m["k"] += int(o["correct"])
    for b, m in bucket_marginals.items():
        m["p"] = m["k"] / m["n"]

    cells: dict[str, dict] = {}
    for o in fit:
        b = feats[o["subj_qid"]]["bucket"]
        key = f"{b}|{o['prop']}"
        c = cells.setdefault(key, {"n": 0, "k": 0, "bucket": b})
        c["n"] += 1
        c["k"] += int(o["correct"])
    for key, c in cells.items():
        backoff = bucket_marginals[c["bucket"]]["p"]
        c["p"] = (c["k"] + M_SHRINK * backoff) / (c["n"] + M_SHRINK)

    def predict(o: dict) -> float:
        f = feats[o["subj_qid"]]
        cell = cells.get(f"{f['bucket']}|{o['prop']}")
        if cell:
            return cell["p"]
        marg = bucket_marginals.get(f["bucket"])
        if marg and marg["n"] >= 10:
            return marg["p"]
        return _logistic_p(w, [math.log1p(f["sitelinks"]), math.log1p(f["n_claims"])])

    auc = _auc([predict(o) for o in ev], [bool(o["correct"]) for o in ev])
    reliable = auc >= AUC_GATE

    profile = {
        "kind": "gcp", "version": 1,
        "model": upstream.model, "upstream": upstream.base_url,
        "evalset": evalset, "n_probes": len(outcomes),
        "bucket_marginals": {str(b): {"n": m["n"], "p": round(m["p"], 4)}
                             for b, m in sorted(bucket_marginals.items())},
        "cells": {k: {"n": c["n"], "p": round(c["p"], 4)} for k, c in cells.items()},
        "logistic_w": [round(x, 6) for x in w],
        "reliability_auc": round(auc, 4), "auc_gate": AUC_GATE,
        "reliable": reliable,
    }
    out.write_text(json.dumps(profile, indent=2))
    if not quiet:
        print(f"\nungrounded accuracy by corpus bucket (0 = head, 7 = tail):")
        for b, m in sorted(bucket_marginals.items()):
            print(f"  bucket {b}: {m['p']:.3f}  (n={m['n']})")
        print(f"\nreliability AUC {auc:.3f} vs gate {AUC_GATE} -> "
              f"{'RELIABLE' if reliable else 'NOT RELIABLE (advisor keeps generic ordering)'}")
        print(f"wrote {out}")
    return profile


def advise_level(profile: dict, store_dir: Path, budget_bytes: int | None,
                 quiet: bool = False) -> int:
    """Recommend a level: the deepest bucket whose marginal competence still
    leaves room for grounding to help (model accuracy < 0.9), capped by the
    byte budget against the installed bucket file sizes."""
    marg = {int(b): m["p"] for b, m in profile["bucket_marginals"].items()}
    level = 0
    spent = 0
    for b in range(8):
        f = store_dir / f"bucket_{b}.sqlite"
        if not f.exists():
            continue  # never recommend past what is installed
        size = f.stat().st_size
        if budget_bytes is not None and spent + size > budget_bytes:
            break
        if marg.get(b, 0.0) >= 0.9:
            continue  # model already knows this region; bytes better spent deeper
        spent += size
        level = b
    if not quiet:
        msg = f"recommended level: L{level}"
        if budget_bytes is not None:
            msg += f" ({spent / 1e6:.0f} MB of {budget_bytes / 1e6:.0f} MB budget)"
        print(msg)
    return level
