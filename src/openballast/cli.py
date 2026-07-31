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
                              help="corpus level L0 (36 MB) .. L7 (full)"),
):
    """Download the serving corpus from Hugging Face (nested — only new buckets fetch)."""
    typer.echo(f"pulling Ballast T0 up to L{level} -> {data_dir()}")
    do_pull(level)
    typer.echo(f"done. installed level: L{installed_level()}")


@app.command()
def lookup(
    question: str = typer.Argument(..., help="a factual question"),
    level: Optional[int] = typer.Option(None, "--level", "-l", help="cap the corpus level"),
):
    """Ground one question and print the evidence blocks (smoke test)."""
    from .store import Store

    store = Store.open(data_dir(), level)
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
    no_mcp: bool = typer.Option(False, help="proxy only"),
):
    """Run the grounding proxy and the MCP server."""
    from . import mcp_server, proxy

    if not no_mcp:
        t = threading.Thread(
            target=mcp_server.run_http,
            kwargs={"level": level, "host": host, "port": mcp_port},
            daemon=True,
        )
        t.start()
        typer.echo(f"MCP (streamable-http): http://{host}:{mcp_port}/mcp")
    typer.echo(f"proxy: http://{host}:{proxy_port}/v1 -> {upstream}")
    typer.echo("point your client's base URL at the proxy; chat requests get grounded.")
    proxy.run(upstream=upstream, host=host, port=proxy_port, level=level)


@app.command()
def mcp(
    level: Optional[int] = typer.Option(None, "--level", "-l", help="cap the corpus level"),
):
    """MCP server on stdio — for Claude Desktop / LM Studio / Cline configs."""
    from . import mcp_server

    mcp_server.run_stdio(level=level)


@app.command()
def status():
    """Installed corpus levels and sizes."""
    d = data_dir()
    lvl = installed_level(d)
    if lvl is None:
        typer.echo(f"no corpus at {d} — run `ballast pull`")
        raise typer.Exit(1)
    typer.echo(f"corpus: {d}")
    typer.echo(f"installed level: L{lvl}")
    for p in sorted(d.glob("*.sqlite")):
        typer.echo(f"  {p.name:24s} {p.stat().st_size / 1e6:8.1f} MB")
    typer.echo(f"openballast {__version__}")


if __name__ == "__main__":
    app()
