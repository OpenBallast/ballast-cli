"""Streaming passthrough: proxied SSE bytes identical to upstream, grounded body
reaches the upstream, non-chat routes pass through untouched."""

from __future__ import annotations

import json
import socket
import threading
import time
from pathlib import Path

import httpx
import pytest
import uvicorn
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, StreamingResponse
from starlette.routing import Route
from starlette.testclient import TestClient

from openballast.proxy import build_app

SSE_CHUNKS = [
    b'data: {"choices":[{"delta":{"content":"Cam"}}]}\n\n',
    b'data: {"choices":[{"delta":{"content":"bridge"}}]}\n\n',
    b"data: [DONE]\n\n",
]

received: dict = {}


def fake_upstream_app() -> Starlette:
    async def chat(request: Request):
        received["body"] = json.loads(await request.body())

        async def gen():
            for c in SSE_CHUNKS:
                yield c

        return StreamingResponse(gen(), media_type="text/event-stream")

    async def tags(_: Request):
        return JSONResponse({"models": ["fake"]})

    return Starlette(routes=[
        Route("/v1/chat/completions", chat, methods=["POST"]),
        Route("/api/tags", tags, methods=["GET"]),
    ])


@pytest.fixture(scope="module")
def upstream_port():
    port = socket.socket().getsockname()[1] or 0
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    config = uvicorn.Config(fake_upstream_app(), host="127.0.0.1", port=port, log_level="error")
    server = uvicorn.Server(config)
    t = threading.Thread(target=server.run, daemon=True)
    t.start()
    for _ in range(100):
        try:
            httpx.get(f"http://127.0.0.1:{port}/api/tags", timeout=1)
            break
        except httpx.HTTPError:
            time.sleep(0.05)
    yield port
    server.should_exit = True


def test_stream_passthrough_and_grounding(corpus_dir: Path, upstream_port: int):
    app = build_app(f"http://127.0.0.1:{upstream_port}", dir=corpus_dir)
    client = TestClient(app)
    body = {
        "model": "m", "stream": True,
        "messages": [{"role": "user", "content": "Where was Douglas Adams born?"}],
    }
    with client.stream("POST", "/v1/chat/completions", json=body) as resp:
        assert resp.status_code == 200
        raw = b"".join(resp.iter_raw())
    assert raw == b"".join(SSE_CHUNKS)  # byte-identical stream

    sent = received["body"]
    assert sent["messages"][0]["role"] == "system"
    assert "Cambridge" in sent["messages"][0]["content"]
    assert sent["messages"][1]["content"] == "Where was Douglas Adams born?"


def test_non_chat_passthrough(corpus_dir: Path, upstream_port: int):
    app = build_app(f"http://127.0.0.1:{upstream_port}", dir=corpus_dir)
    client = TestClient(app)
    resp = client.get("/api/tags")
    assert resp.status_code == 200
    assert resp.json() == {"models": ["fake"]}


def test_info_route(corpus_dir: Path, upstream_port: int):
    app = build_app(f"http://127.0.0.1:{upstream_port}", dir=corpus_dir)
    client = TestClient(app)
    info = client.get("/ballast").json()
    assert info["name"] == "ballast-proxy"
