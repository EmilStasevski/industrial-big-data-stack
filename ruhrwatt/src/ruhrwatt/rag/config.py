"""Assistant settings from environment variables (all optional)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


@dataclass(frozen=True)
class Settings:
    ollama_url: str = field(default_factory=lambda: _env("OLLAMA_HOST", "http://localhost:11434"))
    # 8B with tool calling: ~5-6 GB VRAM at 4-bit, plenty of room for context on 16 GB.
    model: str = field(default_factory=lambda: _env("RUHRWATT_LLM_MODEL", "qwen3:8b"))
    # Multilingual embeddings (German and English questions), ~1.2 GB.
    embed_model: str = field(default_factory=lambda: _env("RUHRWATT_EMBED_MODEL", "bge-m3"))
    num_ctx: int = field(default_factory=lambda: int(_env("RUHRWATT_LLM_CTX", "8192")))
    # How long Ollama keeps the model in VRAM after the last request (lazy load, then reuse).
    keep_alive: str = field(default_factory=lambda: _env("RUHRWATT_LLM_KEEP_ALIVE", "10m"))
    # "auto": no reasoning trace for qwen3 (faster tool calls); "true"/"false" to force.
    think: str = field(default_factory=lambda: _env("RUHRWATT_LLM_THINK", "auto"))
    timeout_s: float = field(default_factory=lambda: float(_env("RUHRWATT_LLM_TIMEOUT", "180")))
    max_steps: int = 6  # tool-call rounds before the agent must answer

    def think_flag(self) -> bool | None:
        if self.think == "true":
            return True
        if self.think == "false":
            return False
        return False if self.model.startswith("qwen3") else None
