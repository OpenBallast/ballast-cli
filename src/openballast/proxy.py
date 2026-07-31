"""OpenAI-compatible grounding proxy.

Sits between any OpenAI-compatible client and any OpenAI-compatible server
(default upstream: Ollama at http://localhost:11434). Chat-completion requests
get a system message with corpus evidence for entities mentioned in the last
user message; everything else — including the response stream — passes through
byte-for-byte.

This is the no-tool-calling path: it grounds models too small to call tools.
"""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import uvicorn
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response, StreamingResponse
from starlette.routing import Route

from .pull import data_dir
from .store import Store
from .template import system_message

HOP_HEADERS = {
    "connection", "keep-alive", "proxy-authenticate", "proxy-authorization",
    "te", "trailers", "transfer-encoding", "upgrade", "host", "content-length",
}


def _last_user_text(messages: list[dict]) -> str:
    for msg in reversed(messages):
        if msg.get("role") != "user":
            continue
        content = msg.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):  # multimodal parts
            return " ".join(
                p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") == "text"
            )
    return ""


def ground_body(store: Store, body: dict, max_triples: int = 24) -> dict:
    """Inject an evidence system message if the last user message links entities."""
    messages = body.get("messages")
    if not isinstance(messages, list) or not messages:
        return body
    question = _last_user_text(messages)
    if not question:
        return body
    result = store.lookup(question, max_triples=max_triples)
    blocks = [b["text"] for b in result["entities"]]
    if not blocks:
        return body
    grounded = dict(body)
    grounded["messages"] = [
        {"role": "system", "content": system_message(blocks, store.level)}
    ] + messages
    return grounded


def build_app(upstream: str, dir: Path | None = None, level: int | None = None,
              max_triples: int = 24) -> Starlette:
    store = Store.open(dir or data_dir(), level)
    client = httpx.AsyncClient(base_url=upstream.rstrip("/"), timeout=httpx.Timeout(600.0))

    def ground(body: dict) -> dict:
        return ground_body(store, body, max_triples)

    async def forward(request: Request, body_bytes: bytes | None = None) -> Response:
        headers = {k: v for k, v in request.headers.items() if k.lower() not in HOP_HEADERS}
        upstream_req = client.build_request(
            request.method,
            request.url.path + (f"?{request.url.query}" if request.url.query else ""),
            headers=headers,
            content=body_bytes if body_bytes is not None else await request.body(),
        )
        upstream_resp = await client.send(upstream_req, stream=True)
        resp_headers = {
            k: v for k, v in upstream_resp.headers.items() if k.lower() not in HOP_HEADERS
        }

        async def stream():
            try:
                async for chunk in upstream_resp.aiter_raw():
                    yield chunk
            finally:
                await upstream_resp.aclose()

        return StreamingResponse(
            stream(), status_code=upstream_resp.status_code, headers=resp_headers
        )

    async def chat_completions(request: Request) -> Response:
        raw = await request.body()
        try:
            body = json.loads(raw)
            grounded = ground(body)
            raw = json.dumps(grounded).encode()
        except (json.JSONDecodeError, UnicodeDecodeError):
            pass  # malformed body: pass through untouched, upstream decides
        return await forward(request, raw)

    async def passthrough(request: Request) -> Response:
        return await forward(request)

    async def info(_: Request) -> JSONResponse:
        return JSONResponse({
            "name": "ballast-proxy",
            "upstream": upstream,
            "level": store.level,
            "what": "OpenAI-compatible grounding proxy — chat completions get "
                    "corpus evidence injected as a system message",
        })

    return Starlette(routes=[
        Route("/ballast", info, methods=["GET"]),
        Route("/v1/chat/completions", chat_completions, methods=["POST"]),
        Route("/{path:path}", passthrough,
              methods=["GET", "POST", "PUT", "DELETE", "PATCH", "HEAD", "OPTIONS"]),
    ])


def run(upstream: str = "http://localhost:11434", host: str = "127.0.0.1",
        port: int = 11435, dir: Path | None = None, level: int | None = None) -> None:
    uvicorn.run(build_app(upstream, dir, level), host=host, port=port, log_level="warning")
