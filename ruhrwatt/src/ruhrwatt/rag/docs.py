"""Documentation index for the "how / why" questions (LanceDB + Ollama embeddings).

* Corpus: ``README.md`` and ``docs/*.md`` (methods, definitions, caveats). Numbers are not
  indexed: they come from the data tools, so the index never goes stale on figures.
* Chunks follow the markdown headings; long sections are split at paragraphs.
* Lazy: the index is built once, only new or changed chunks are re-embedded on refresh
  (chunk id = hash of its text), and the table is opened on the first search.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from ruhrwatt import config

MAX_CHARS = 1200
TABLE = "chunks"

Embed = Callable[[list[str]], list[list[float]]]


@dataclass(frozen=True)
class Chunk:
    source: str  # file name
    heading: str  # "Gold layer > Backtest"
    text: str

    @property
    def id(self) -> str:
        return hashlib.sha256(f"{self.source}|{self.heading}|{self.text}".encode()).hexdigest()[:16]

    def for_embedding(self) -> str:
        return f"{self.source} > {self.heading}\n{self.text}"


def project_root() -> Path:
    here = Path(__file__).resolve()
    for p in here.parents:
        if (p / "pyproject.toml").exists():
            return p
    return Path.cwd()


def corpus_files(root: Path | None = None) -> list[Path]:
    root = root or project_root()
    files = [root / "README.md", *sorted((root / "docs").glob("*.md"))]
    return [f for f in files if f.exists()]


def chunk_markdown(text: str, source: str, max_chars: int = MAX_CHARS) -> list[Chunk]:
    """Split by headings, keep the heading path, split long sections at blank lines."""
    chunks: list[Chunk] = []
    path: list[str] = []
    buf: list[str] = []

    def flush() -> None:
        body = "\n".join(buf).strip()
        buf.clear()
        if not body:
            return
        heading = " > ".join(path) or source
        parts, cur = [], ""
        for para in re.split(r"\n\s*\n", body):
            if cur and len(cur) + len(para) + 2 > max_chars:
                parts.append(cur)
                cur = para
            else:
                cur = f"{cur}\n\n{para}" if cur else para
        if cur:
            parts.append(cur)
        chunks.extend(Chunk(source, heading, p.strip()) for p in parts if p.strip())

    in_code = False
    for line in text.splitlines():
        if line.startswith("```"):
            in_code = not in_code
        m = None if in_code else re.match(r"^(#{1,4})\s+(.*)", line)
        if m:
            flush()
            level = len(m.group(1))
            path[:] = [*path[: level - 1], m.group(2).strip()]
        else:
            buf.append(line)
    flush()
    return chunks


def load_chunks(root: Path | None = None) -> list[Chunk]:
    out: list[Chunk] = []
    for f in corpus_files(root):
        out.extend(chunk_markdown(f.read_text(encoding="utf-8"), f.name))
    return out


def index_dir() -> Path:
    return config.data_dir() / "rag" / "docs.lance"


def build_index(embed: Embed, root: Path | None = None, path: Path | None = None) -> dict:
    """Create or refresh the index; only chunks whose text changed are embedded again."""
    import lancedb

    path = path or index_dir()
    path.mkdir(parents=True, exist_ok=True)
    db = lancedb.connect(str(path))
    chunks = load_chunks(root)
    wanted = {c.id: c for c in chunks}

    cached: dict[str, list[float]] = {}
    try:
        existing = db.open_table(TABLE)
    except ValueError:  # first build
        existing = None
    if existing is not None:
        for row in existing.to_arrow().select(["id", "vector"]).to_pylist():
            cached[row["id"]] = row["vector"]
    new = [c for c in chunks if c.id not in cached]
    vectors = embed([c.for_embedding() for c in new]) if new else []
    cached.update({c.id: v for c, v in zip(new, vectors, strict=True)})

    rows = [
        {
            "id": c.id,
            "source": c.source,
            "heading": c.heading,
            "text": c.text,
            "vector": cached[c.id],
        }
        for c in wanted.values()
    ]
    db.create_table(TABLE, data=rows, mode="overwrite")
    return {
        "chunks": len(rows),
        "embedded": len(new),
        "reused": len(rows) - len(new),
        "path": str(path),
    }


class DocSearch:
    """Opens the index on first use (lazy) and embeds queries with the same model."""

    def __init__(self, embed: Embed, path: Path | None = None):
        self.embed = embed
        self.path = path or index_dir()
        self._table = None

    def _open(self):
        if self._table is None:
            import lancedb

            if not self.path.exists():
                raise FileNotFoundError(f"docs index missing at {self.path}; run `make rag-index`")
            self._table = lancedb.connect(str(self.path)).open_table(TABLE)
        return self._table

    def search(self, query: str, k: int = 4) -> dict:
        """Tool entry point: the k most similar doc chunks."""
        k = max(1, min(int(k or 4), 8))
        try:
            table = self._open()
        except FileNotFoundError as exc:
            return {"error": str(exc)}
        vec = self.embed([query])[0]
        hits = table.search(vec).metric("cosine").limit(k).to_list()
        return {
            "results": [
                {
                    "source": f"{h['source']} > {h['heading']}",
                    "text": h["text"],
                    "similarity": round(1 - float(h["_distance"]), 3),
                }
                for h in hits
            ],
            "source": "project docs: " + "; ".join(f"{h['source']} > {h['heading']}" for h in hits),
        }


def search_docs_tool(searcher: DocSearch) -> Callable[..., dict]:
    def search_docs(query: str, k: int = 4) -> dict:
        """Search the project documentation."""
        return searcher.search(query, k)

    return search_docs
