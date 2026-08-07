"""Download a Ballast serving artifact from Hugging Face.

Tiers are independent corpora, each with its own repo and local directory:
t0 (Wikidata facts), t1 (full-body Wikipedia passages), t2 (OpenStax
textbook passages). Levels are nested: Lk needs properties.sqlite +
bucket_0..k. Upgrading a level downloads only the missing bucket files;
nothing already present is re-fetched.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

from huggingface_hub import hf_hub_download

TIER_REPOS = {
    "t0": "OpenBallast/ballast-t0",
    "t1": "OpenBallast/ballast-t1",
    "t2": "OpenBallast/ballast-t2",
}
REPO_ID = TIER_REPOS["t0"]  # kept for back-compat imports
SERVING_PREFIX = "serving/sqlite"
MAX_BUCKET = 7


def ballast_home() -> Path:
    env = os.environ.get("BALLAST_HOME")
    if env:
        return Path(env)
    return Path.home() / ".ballast"


def data_dir(corpus: str = "t0") -> Path:
    return ballast_home() / corpus


def corpora() -> list[str]:
    """Installed corpus names (any dir under BALLAST_HOME with bucket DBs)."""
    home = ballast_home()
    if not home.exists():
        return []
    return sorted(
        d.name for d in home.iterdir()
        if d.is_dir() and any(d.glob("bucket_*.sqlite"))
    )


def _fetch(filename: str, dest: Path, repo_id: str) -> Path:
    """Download `<filename>.zst` from the dataset and decompress to `<filename>`."""
    import zstandard

    cached = hf_hub_download(
        repo_id=repo_id, repo_type="dataset",
        filename=f"{SERVING_PREFIX}/{filename}.zst",
        local_dir=dest / "_hf",
    )
    target = dest / filename
    tmp = dest / (filename + ".part")
    dctx = zstandard.ZstdDecompressor()
    with open(cached, "rb") as src, open(tmp, "wb") as out:
        dctx.copy_stream(src, out)
    tmp.replace(target)
    Path(cached).unlink(missing_ok=True)
    return target


def _print_size_preview(repo_id: str, wanted: list[str], quiet: bool) -> None:
    """If the repo publishes serving_manifest.json, show what the pull costs
    before any download starts. Absence is fine — sizes then print per file."""
    if quiet:
        return
    try:
        path = hf_hub_download(repo_id=repo_id, repo_type="dataset",
                               filename=f"{SERVING_PREFIX}/serving_manifest.json")
        manifest = json.loads(Path(path).read_text(encoding="utf-8"))
        files = manifest.get("files", {})
        down = sum(files[n]["download_bytes"] for n in wanted if n in files)
        disk = sum(files[n]["disk_bytes"] for n in wanted if n in files)
        if down:
            print(f"  total: {down / 1e6:.0f} MB download -> "
                  f"{disk / 1e9:.1f} GB on disk")
    except Exception:
        pass


def pull(level: int, tier: str = "t0", quiet: bool = False) -> Path:
    if tier not in TIER_REPOS:
        raise ValueError(f"unknown tier '{tier}' (have: {', '.join(TIER_REPOS)})")
    repo_id = TIER_REPOS[tier]
    level = max(0, min(level, MAX_BUCKET))
    dest = data_dir(tier)
    dest.mkdir(parents=True, exist_ok=True)

    wanted = ["properties.sqlite"] + [f"bucket_{b}.sqlite" for b in range(level + 1)]
    _print_size_preview(repo_id, wanted, quiet)
    for name in wanted:
        target = dest / name
        if target.exists():
            if not quiet:
                print(f"  {name}: already present ({target.stat().st_size / 1e6:.0f} MB)")
            continue
        t0 = time.time()
        if not quiet:
            print(f"  {name}: downloading...", flush=True)
        _fetch(name, dest, repo_id)
        if not quiet:
            size = (dest / name).stat().st_size / 1e6
            print(f"  {name}: {size:.0f} MB in {time.time() - t0:.0f}s")

    manifest = {
        "repo": repo_id,
        "level": max(level, installed_level(dest) or 0),
        "files": {p.name: p.stat().st_size for p in dest.glob("*.sqlite")},
    }
    (dest / "installed.json").write_text(json.dumps(manifest, indent=2))
    return dest


def installed_level(dest: Path | None = None) -> int | None:
    dest = dest or data_dir()
    buckets = [b for b in range(MAX_BUCKET + 1) if (dest / f"bucket_{b}.sqlite").exists()]
    if not buckets or not (dest / "properties.sqlite").exists():
        return None
    # level = highest contiguous bucket from 0
    level = -1
    for b in range(MAX_BUCKET + 1):
        if b in buckets:
            level = b
        else:
            break
    return level if level >= 0 else None
