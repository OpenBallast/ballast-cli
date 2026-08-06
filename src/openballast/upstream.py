"""Minimal OpenAI-compatible completion client for `profile` and `eval`.

Talks to the same upstreams `serve` proxies (Ollama, LM Studio, llama.cpp
server, vLLM) — the public tooling never loads model weights itself, so the
package stays GPU-free.
"""

from __future__ import annotations

import httpx


class Upstream:
    def __init__(self, base_url: str = "http://localhost:11434",
                 model: str = "", timeout: float = 120.0):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self._client = httpx.Client(timeout=timeout)

    def complete(self, prompt: str, max_tokens: int = 512) -> str:
        """One deterministic completion (temperature 0).

        The budget is generous because reasoning-mode models (Qwen3 family
        and friends) spend tokens thinking before the answer; the caller
        grades only the extracted answer line, so verbosity is harmless but
        a starved budget silently produces empty answers.
        """
        r = self._client.post(
            f"{self.base_url}/v1/chat/completions",
            json={
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
                "temperature": 0,
            },
        )
        r.raise_for_status()
        data = r.json()
        return (data["choices"][0]["message"]["content"] or "").strip()

    def close(self) -> None:
        self._client.close()
