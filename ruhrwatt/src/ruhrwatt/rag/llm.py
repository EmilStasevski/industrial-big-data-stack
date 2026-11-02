"""Minimal Ollama client (httpx): chat with tool calls, embeddings, health checks.

Plain HTTP instead of an SDK keeps the dependency list short and makes the client easy to
test with ``httpx.MockTransport``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import httpx

from ruhrwatt.rag.config import Settings


class OllamaError(RuntimeError):
    """Ollama is not reachable, a model is missing, or a request failed."""


@dataclass
class ChatReply:
    content: str
    tool_calls: list[dict[str, Any]] = field(default_factory=list)  # {"name", "arguments"}
    thinking: str = ""
    eval_tokens: int = 0
    seconds: float = 0.0
    raw: dict[str, Any] = field(default_factory=dict)


class OllamaClient:
    def __init__(
        self, settings: Settings | None = None, transport: httpx.BaseTransport | None = None
    ):
        self.s = settings or Settings()
        self.http = httpx.Client(
            base_url=self.s.ollama_url, timeout=self.s.timeout_s, transport=transport
        )

    # ------------------------------------------------------------------ plumbing

    def _post(self, path: str, body: dict) -> dict:
        try:
            r = self.http.post(path, json=body)
        except httpx.TransportError as exc:
            raise OllamaError(
                f"Ollama is not reachable at {self.s.ollama_url} ({exc}). "
                "Start it with `ollama serve` (or the Ollama app)."
            ) from exc
        if r.status_code == 404 and "not found" in r.text:
            model = body.get("model", "?")
            raise OllamaError(f"model {model!r} is not installed: run `ollama pull {model}`")
        if r.status_code >= 400:
            raise OllamaError(f"Ollama {path} failed: HTTP {r.status_code}: {r.text[:300]}")
        return r.json()

    def _get(self, path: str) -> dict:
        try:
            r = self.http.get(path)
            r.raise_for_status()
        except httpx.TransportError as exc:
            raise OllamaError(
                f"Ollama is not reachable at {self.s.ollama_url}. Start it with `ollama serve`."
            ) from exc
        return r.json()

    # ------------------------------------------------------------------ API

    def version(self) -> str:
        return self._get("/api/version").get("version", "?")

    def installed_models(self) -> list[str]:
        return [m["name"] for m in self._get("/api/tags").get("models", [])]

    def loaded_models(self) -> list[dict]:
        """Models currently in memory, with ``size_vram`` (bytes on the GPU)."""
        return self._get("/api/ps").get("models", [])

    def chat(self, messages: list[dict], tools: list[dict] | None = None) -> ChatReply:
        body: dict[str, Any] = {
            "model": self.s.model,
            "messages": messages,
            "stream": False,
            "keep_alive": self.s.keep_alive,
            "options": {"temperature": 0, "num_ctx": self.s.num_ctx},
        }
        if tools:
            body["tools"] = tools
        think = self.s.think_flag()
        if think is not None:
            body["think"] = think
        data = self._post("/api/chat", body)
        msg = data.get("message", {})
        calls = []
        for c in msg.get("tool_calls") or []:
            fn = c.get("function", {})
            args = fn.get("arguments", {})
            if isinstance(args, str):  # some models return a JSON string
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {"_raw": args}
            calls.append({"name": fn.get("name", ""), "arguments": args or {}})
        return ChatReply(
            content=msg.get("content", "") or "",
            tool_calls=calls,
            thinking=msg.get("thinking", "") or "",
            eval_tokens=int(data.get("eval_count", 0) or 0),
            seconds=float(data.get("total_duration", 0) or 0) / 1e9,
            raw=data,
        )

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        data = self._post(
            "/api/embed",
            {"model": self.s.embed_model, "input": texts, "keep_alive": self.s.keep_alive},
        )
        vecs = data.get("embeddings")
        if not vecs or len(vecs) != len(texts):
            raise OllamaError("unexpected /api/embed response")
        return vecs
