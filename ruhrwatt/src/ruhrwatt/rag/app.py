"""Wiring: Ollama client + doc search + data tools -> agent."""

from __future__ import annotations

from ruhrwatt.rag import docs, tools
from ruhrwatt.rag.agent import Agent
from ruhrwatt.rag.config import Settings
from ruhrwatt.rag.llm import OllamaClient


def make_agent(settings: Settings | None = None) -> Agent:
    s = settings or Settings()
    client = OllamaClient(s)
    searcher = docs.DocSearch(client.embed)  # opens the index lazily, on first search
    return Agent(client, tools.registry(docs.search_docs_tool(searcher)), s.max_steps)


def doctor(settings: Settings | None = None) -> list[tuple[str, bool, str]]:
    """Checks for `ruhrwatt rag doctor`: (name, ok, detail)."""
    from ruhrwatt import config
    from ruhrwatt.rag.llm import OllamaError

    s = settings or Settings()
    c = OllamaClient(s)
    out: list[tuple[str, bool, str]] = []
    try:
        out.append(("ollama", True, f"version {c.version()} at {s.ollama_url}"))
        have = c.installed_models()
    except OllamaError as exc:
        return [("ollama", False, str(exc))]

    def installed(m: str) -> bool:
        return m in have or f"{m}:latest" in have

    for role, m in (("chat model", s.model), ("embedding model", s.embed_model)):
        ok = installed(m)
        out.append((role, ok, m if ok else f"{m} missing: run `ollama pull {m}`"))
    gold = config.data_dir() / "gold" / "market_monthly.parquet"
    out.append(
        ("gold tables", gold.exists(), str(gold) if gold.exists() else "run `make pipeline`")
    )
    idx = docs.index_dir()
    out.append(("docs index", idx.exists(), str(idx) if idx.exists() else "run `make rag-index`"))
    for m in c.loaded_models():
        gb = m.get("size_vram", 0) / 1e9
        total = m.get("size", 0) / 1e9
        out.append((f"in VRAM: {m.get('name')}", True, f"{gb:.1f} GB on GPU of {total:.1f} GB"))
    return out
