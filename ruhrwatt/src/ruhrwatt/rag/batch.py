"""Run a JSON file of questions in one go and write a report to paste or attach.

File format::

    {"questions": [
        {"id": "basis_monat", "category": "einfach", "question": "..."},
        {"id": "folge", "category": "folgefragen", "conversation": ["...", "...", "..."]}
    ]}

Each item gets a fresh agent; the turns of a ``conversation`` share one chat history, so
follow-ups ("Und welche waren am teuersten?") are tested too.
"""

from __future__ import annotations

import json
import traceback
from collections.abc import Callable
from pathlib import Path


def load_questions(path: Path) -> list[dict]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    items = data["questions"] if isinstance(data, dict) else data
    seen: set[str] = set()
    for i, it in enumerate(items):
        if "id" not in it:
            raise ValueError(f"item {i} has no 'id'")
        if it["id"] in seen:
            raise ValueError(f"duplicate id {it['id']!r}")
        seen.add(it["id"])
        if not it.get("question") and not it.get("conversation"):
            raise ValueError(f"item {it['id']!r} needs 'question' or 'conversation'")
    return items


def run(
    make_agent: Callable[[], object],
    items: list[dict],
    progress: Callable[[dict], None] | None = None,
    fatal: tuple[type[Exception], ...] = (),
) -> list[dict]:
    """One result per turn. ``fatal`` errors (e.g. Ollama down) stop the run; others are kept."""
    results = []
    for it in items:
        turns = it.get("conversation") or [it["question"]]
        agent = make_agent()
        for n, q in enumerate(turns, 1):
            rid = it["id"] if len(turns) == 1 else f"{it['id']}.{n}"
            row = {"id": rid, "category": it.get("category", ""), "question": q}
            try:
                ans = agent.ask(q)
                row |= {
                    "answer": ans.text,
                    "tools": [
                        {"name": c["name"], "arguments": c["arguments"]} for c in ans.tool_calls
                    ],
                    "tool_errors": [
                        c["result"]["error"] for c in ans.tool_calls if "error" in c["result"]
                    ],
                    "sources": ans.sources,
                    "ungrounded": ans.ungrounded,
                    "steps": ans.steps,
                    "stopped": ans.stopped,
                    "seconds": round(ans.seconds, 1),
                }
            except fatal:
                raise
            except Exception as exc:  # keep going, record the failure
                row |= {"answer": "", "error": f"{type(exc).__name__}: {exc}",
                        "trace": traceback.format_exc(limit=3)}  # fmt: skip
            results.append(row)
            if progress:
                progress(row)
    return results


def to_markdown(results: list[dict], model: str) -> str:
    total = sum(r.get("seconds", 0) for r in results)
    lines = [
        f"# Ruhrwatt batch: {model}",
        "",
        f"{len(results)} questions, {total:.0f} s total, "
        f"{sum(1 for r in results if r.get('tools'))} with tool calls, "
        f"{sum(1 for r in results if r.get('error'))} errors, "
        f"{sum(1 for r in results if r.get('ungrounded'))} with unsupported numbers",
        "",
    ]
    for r in results:
        lines.append(f"## {r['id']} ({r['category']})")
        lines.append(f"**Q:** {r['question']}")
        lines.append("")
        if r.get("error"):
            lines.append(f"**ERROR:** {r['error']}")
        else:
            lines.append(r["answer"])
            calls = "; ".join(
                f"{t['name']}({json.dumps(t['arguments'], ensure_ascii=False)})" for t in r["tools"]
            )
            lines.append("")
            lines.append(f"_tools: {calls or 'none'} | {r['seconds']} s | {r['stopped']}_")
            if r["tool_errors"]:
                lines.append(f"_tool errors: {' | '.join(r['tool_errors'])}_")
            if r.get("ungrounded"):
                nums = ", ".join(f"{n:g}" for n in r["ungrounded"])
                lines.append(f"_numbers not found in any tool result: {nums}_")
        lines.append("")
    return "\n".join(lines)
