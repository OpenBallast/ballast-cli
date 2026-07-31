"""MCP server over the local Ballast store.

Same tool contract as the hosted demo endpoint (mcp.openballast.org):
resolve / evidence / lookup. docs/mcp.md applies to both.

stdio:            ballast mcp
streamable-http:  part of `ballast serve` (default :11436)
"""

from __future__ import annotations

import json
from pathlib import Path

from mcp.server.mcpserver import MCPServer

from .pull import data_dir
from .store import Store

INSTRUCTIONS = (
    "Ballast T0: 25M-entity CC0 knowledge corpus, served locally. "
    "Call `lookup` with the user's question before answering factual queries; "
    "prepend the returned evidence blocks to your prompt."
)


def build_server(dir: Path | None = None, level: int | None = None) -> MCPServer:
    store = Store.open(dir or data_dir(), level)
    server = MCPServer(name="ballast", instructions=INSTRUCTIONS)

    @server.tool()
    def resolve(name: str, limit: int = 5) -> str:
        """Resolve an entity name to Wikidata Q-ids in the Ballast T0 corpus
        (normalized label/alias match, most-notable first)."""
        hits = store.resolve(name, limit)
        return json.dumps({"hits": [vars(h) for h in hits]}, indent=1)

    @server.tool()
    def evidence(id: str, max_triples: int = 32) -> str:
        """Grounding facts for one entity (Q-id like Q42, or a name) as a compact
        evidence block. Facts whose object entity falls outside the installed
        corpus level are dropped, exactly like the truncated artifact."""
        ev = store.evidence(id, max_triples=max_triples)
        return json.dumps(vars(ev), indent=1)

    @server.tool()
    def lookup(question: str, max_triples: int = 24) -> str:
        """One-shot grounding for a whole question: mines capitalized entity
        mentions, resolves each against the corpus, returns evidence blocks for
        every hit. Feed the blocks to your model before answering."""
        return json.dumps(store.lookup(question, max_triples=max_triples), indent=1)

    return server


def run_stdio(dir: Path | None = None, level: int | None = None) -> None:
    build_server(dir, level).run(transport="stdio")


def run_http(dir: Path | None = None, level: int | None = None,
             host: str = "127.0.0.1", port: int = 11436) -> None:
    build_server(dir, level).run(transport="streamable-http", host=host, port=port)
