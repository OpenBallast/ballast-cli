"""ballast — pull a quantized knowledge corpus, ground any local model."""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Optional

import typer

from . import __version__
from .pull import MAX_BUCKET, data_dir, installed_level
from .pull import pull as do_pull

app = typer.Typer(no_args_is_help=True, add_completion=False, help=__doc__)


@app.command()
def pull(
    level: int = typer.Option(3, "--level", "-l", min=0, max=MAX_BUCKET,
                              help="corpus level L0 .. L7 (nested; sizes print before download)"),
    tier: str = typer.Option("t0", "--tier", "-t",
                             help="t0 = Wikidata facts, t1 = Wikipedia passages, t2 = OpenStax passages"),
):
    """Download a serving corpus from Hugging Face (nested — only new buckets fetch)."""
    typer.echo(f"pulling Ballast {tier.upper()} up to L{level} -> {data_dir(tier)}")
    do_pull(level, tier=tier)
    typer.echo(f"done. installed level: L{installed_level(data_dir(tier))}")
    if tier != "t0":
        typer.echo(f"serve it with: ballast serve --corpus {tier}")


@app.command()
def build(
    src: Path = typer.Argument(..., exists=True, file_okay=False,
                               help="directory of .md/.txt/.parquet documents"),
    name: str = typer.Option("byo", "--name", "-n", help="corpus name (serve with --corpus NAME)"),
    out: Optional[Path] = typer.Option(None, help="output dir (default BALLAST_HOME/<name>)"),
    chunk_bytes: int = typer.Option(1200, help="passage chunk byte budget"),
):
    """Build a servable corpus from your own documents (bring-your-own corpus).

    Parquet inputs need a `text` column; `title` and `rank` (0..1, 1 = most
    important) are optional. Without ranks every document lands in one level.
    """
    from .build import build as do_build

    do_build(src, name=name, out=out, chunk_bytes=chunk_bytes)
    typer.echo(f"try: ballast lookup --corpus {name} \"<a question about your docs>\"")


@app.command()
def profile(
    model: str = typer.Option(..., "--model", "-m", help="upstream model name (e.g. qwen3.5:9b)"),
    upstream: str = typer.Option("http://localhost:11434", help="OpenAI-compatible upstream"),
    evalset: str = typer.Option("matrix", help="evalset name or local probes parquet"),
    limit: int = typer.Option(2000, help="probe sample size"),
    budget: Optional[str] = typer.Option(None, help="corpus byte budget, e.g. 2GB — emits a level recommendation"),
    corpus: str = typer.Option("t0", "--corpus", help="installed corpus to profile against"),
    level: Optional[int] = typer.Option(None, "--level", "-l"),
    out: Optional[Path] = typer.Option(None, help="output .gcp.json path"),
):
    """Profile a local model: where does its parametric knowledge run out?

    Writes a grounding competence profile (.gcp.json) with per-region
    accuracy and a reliability AUC against the 0.58 gate, plus an optional
    budget-aware corpus level recommendation.
    """
    from .profile import advise_level, run_profile
    from .store import Store
    from .upstream import Upstream

    store = Store.open(data_dir(corpus), level)
    up = Upstream(upstream, model=model)
    safe = model.replace("/", "__").replace(":", "_")
    out_path = out or Path(f"{safe}.gcp.json")
    prof = run_profile(up, store, evalset, limit, out_path)
    if budget:
        m = {"kb": 1e3, "mb": 1e6, "gb": 1e9}
        b = budget.strip().lower()
        for suffix, mult in m.items():
            if b.endswith(suffix):
                advise_level(prof, data_dir(corpus), int(float(b[: -len(suffix)]) * mult))
                break
        else:
            advise_level(prof, data_dir(corpus), int(float(b)))


@app.command()
def eval(
    model: str = typer.Option(..., "--model", "-m", help="upstream model name"),
    upstream: str = typer.Option("http://localhost:11434", help="OpenAI-compatible upstream"),
    evalset: str = typer.Option("matrix", help="evalset name or local probes parquet"),
    limit: int = typer.Option(500, help="probe sample size"),
    corpus: str = typer.Option("t0", "--corpus", help="installed corpus to evaluate"),
    level: Optional[int] = typer.Option(None, "--level", "-l"),
    outdir: Path = typer.Option(Path("ballast_eval"), help="arm checkpoints + summary land here"),
):
    """Run the three-arm instrument: ungrounded / realized / saturated.

    Reports the knowledge-limited band (S - U) and the delivery ratio
    (R - U) / (S - U) — the fraction of the reachable gap retrieval closes.
    Interrupted runs resume; finished arms are never re-queried.
    """
    from .instrument import run_eval
    from .store import Store
    from .upstream import Upstream

    store = Store.open(data_dir(corpus), level)
    up = Upstream(upstream, model=model)
    run_eval(up, store, evalset, limit, outdir)


@app.command()
def lookup(
    question: str = typer.Argument(..., help="a factual question"),
    level: Optional[int] = typer.Option(None, "--level", "-l", help="cap the corpus level"),
    corpus: str = typer.Option("t0", "--corpus", help="corpus name (t0 or a built one)"),
):
    """Ground one question and print the evidence blocks (smoke test)."""
    from .store import Store

    store = Store.open(data_dir(corpus), level)
    result = store.lookup(question)
    if not result["entities"]:
        typer.echo(f"(no corpus entities linked at L{store.level})")
        raise typer.Exit(1)
    for block in result["entities"]:
        typer.echo(f"[{block['mention']} -> {block['qid']} @ L{block['level']}]")
        typer.echo(block["text"])
        typer.echo("")


@app.command()
def serve(
    upstream: str = typer.Option("http://localhost:11434", help="OpenAI-compatible upstream (Ollama default)"),
    host: str = typer.Option("127.0.0.1"),
    proxy_port: int = typer.Option(11435, help="OpenAI-compatible grounding proxy port"),
    mcp_port: int = typer.Option(11436, help="MCP streamable-http port"),
    level: Optional[int] = typer.Option(None, "--level", "-l", help="cap the corpus level"),
    corpus: str = typer.Option("t0", "--corpus", help="corpus name (t0 or a built one)"),
    no_mcp: bool = typer.Option(False, help="proxy only"),
):
    """Run the grounding proxy and the MCP server."""
    from . import mcp_server, proxy

    corpus_dir = data_dir(corpus)
    if not no_mcp:
        t = threading.Thread(
            target=mcp_server.run_http,
            kwargs={"level": level, "host": host, "port": mcp_port, "dir": corpus_dir},
            daemon=True,
        )
        t.start()
        typer.echo(f"MCP (streamable-http): http://{host}:{mcp_port}/mcp")
    typer.echo(f"proxy: http://{host}:{proxy_port}/v1 -> {upstream}")
    typer.echo("point your client's base URL at the proxy; chat requests get grounded.")
    proxy.run(upstream=upstream, host=host, port=proxy_port, level=level, dir=corpus_dir)


@app.command()
def mcp(
    level: Optional[int] = typer.Option(None, "--level", "-l", help="cap the corpus level"),
    corpus: str = typer.Option("t0", "--corpus", help="corpus name (t0 or a built one)"),
):
    """MCP server on stdio — for Claude Desktop / LM Studio / Cline configs."""
    from . import mcp_server

    mcp_server.run_stdio(level=level, dir=data_dir(corpus))


@app.command()
def status():
    """Installed corpora, levels, and sizes."""
    from .pull import ballast_home, corpora

    names = corpora()
    if not names:
        typer.echo(f"no corpora under {ballast_home()} — run `ballast pull` or `ballast build`")
        raise typer.Exit(1)
    for name in names:
        d = data_dir(name)
        lvl = installed_level(d)
        typer.echo(f"corpus: {name}  ({d})")
        if lvl is not None:
            typer.echo(f"  installed level: L{lvl}")
        for p in sorted(d.glob("*.sqlite")):
            typer.echo(f"  {p.name:24s} {p.stat().st_size / 1e6:8.1f} MB")
    typer.echo(f"openballast {__version__}")


if __name__ == "__main__":
    app()
