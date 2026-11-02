"""Tool-calling loop: the model asks for data, we run the tools, it writes the answer.

Guardrails:

* the system prompt forbids numbers that did not come from a tool result;
* at most ``max_steps`` tool rounds, then the model must answer with what it has;
* every tool result carries a ``source``; the answer lists the sources actually used.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Protocol

from ruhrwatt.rag import tools as T
from ruhrwatt.rag.llm import ChatReply
from ruhrwatt.rag.numbers import numbers

SYSTEM = """You are Ruhrwatt, an energy market analyst for industry in North Rhine-Westphalia (NRW).
Your data: SMARD (Bundesnetzagentur) day-ahead prices for the DE-LU bidding zone, plus
load, generation and TSO forecasts for Germany and the Amprion control area (the NRW proxy).
Today is {today}. {coverage}

Facts about this data (verified, use them):
- Germany and Luxembourg form ONE bidding zone (DE-LU): the day-ahead price is the same in
  NRW as anywhere else in Germany. Regions differ in generation mix and load, not in price.
- SMARD has no NRW-level data; the Amprion control area covers most of NRW and is the proxy.
- Clock changes: the day the clocks go back (last Sunday of October) has 25 hours (100
  quarter-hours), the spring day 23 hours (92). Timestamps are stored in UTC.
- Day-ahead prices are 15-minute values from 2025-10-01; before, hourly prices repeated 4x.
- The price model uses the TSOs' day-ahead wind/PV forecasts for the delivery day. TSOs may
  publish them after the 12:00 auction (by 18:00 the day before), so the backtest is slightly
  optimistic; traders have similar weather forecasts before noon. Details: search_docs.

Rules:
1. Every number you state must come from a tool result in this conversation. Never use
   numbers from memory. If you need a figure, call a tool. Do not do arithmetic yourself
   when a tool can return the figure (costs: flat_mw; week/day rankings: period_breakdown).
2. If the data cannot answer the question (outside the date range, not in the data), say so
   plainly instead of guessing.
3. Any question about this project's data, methods, models, forecasts, the Amprion area,
   time zones or caveats: call search_docs first and answer from what it returns, not from
   general knowledge.
4. Call tools directly; never describe a tool call you are going to make. To compare two
   periods, call price_stats once per period. When a tool result has a "warning", repeat it.
5. When search_docs returns text, base the answer on that text, not on general knowledge.
6. Relative periods ("last 3 months", "letzte Woche", "gestern", "im Oktober") ARE periods:
   turn them into start/end dates from today's date, pass them to the tool, and name the
   dates in the answer. Only if the user names no period at all, use the full history
   (omit start/end) and say so.
   Rankings: months via monthly_market(sort_by=...), weeks/days via period_breakdown.
   Two-part questions ("the most expensive month, and its peak/off-peak gap"): rank by the
   first criterion only (price_base), then read the second figure from that same row.
7. Units: prices in EUR/MWh, energy in MWh. Load-shifting savings are an upper bound on the
   day-ahead energy cost only; say so when you report them.
8. Answer in the language of the question (German or English), concisely, and name the
   period the numbers refer to."""


class Chat(Protocol):
    def chat(self, messages: list[dict], tools: list[dict] | None = None) -> ChatReply: ...


@dataclass
class Answer:
    text: str
    sources: list[str] = field(default_factory=list)
    tool_calls: list[dict[str, Any]] = field(default_factory=list)  # name, arguments, result
    steps: int = 0
    seconds: float = 0.0
    stopped: str = "answered"  # or "max_steps"
    ungrounded: list[float] = field(default_factory=list)  # numbers no tool result supports


def system_prompt(tools: dict[str, T.Tool]) -> str:
    coverage = ""
    if "dataset_info" in tools:
        info = T.call(tools, "dataset_info", {})
        if "error" not in info:
            coverage = (
                f"Data covers {info['first_day']} to {info['last_complete_quarter_hour_berlin']} "
                f"(prices published up to {info['last_published_price_berlin']})."
            )
    return SYSTEM.format(today=date.today().isoformat(), coverage=coverage)


NUDGE = (
    "You answered without calling any tool. If the question concerns prices, plants, the "
    "data or this project's methods, call the matching tool now (search_docs for methods "
    "and definitions). Only if no tool can help, repeat your answer unchanged."
)
NUDGE_ANNOUNCED = "Do not describe the tool call. Call {name} now."


NUDGE_GROUNDING = (
    "These numbers in your answer are not in any tool result: {nums}. Fetch them with a "
    "tool, or remove them. Do not estimate or invent figures."
)
_DATE_TIME = re.compile(r"\d{4}-\d{2}-\d{2}(?:[ T]\d{2}:\d{2}(?::\d{2})?Z?)?|\b\d{1,2}:\d{2}\b")


def _numbers_in(obj: Any) -> list[float]:
    """All numbers in a tool result (values and numbers inside strings)."""
    if isinstance(obj, bool):
        return []
    if isinstance(obj, int | float):
        return [float(obj)]
    if isinstance(obj, str):
        return numbers(_DATE_TIME.sub(" ", obj))
    if isinstance(obj, dict):
        return [n for v in obj.values() for n in _numbers_in(v)]
    if isinstance(obj, list):
        return [n for v in obj for n in _numbers_in(v)]
    return []


def unsupported_numbers(answer: str, question: str, results: list[dict]) -> list[float]:
    """Numbers in the answer that are not in a tool result or the question, nor a simple
    difference or percentage of two of them. Small integers (counts, list numbering) and
    dates/times are ignored."""
    known = {abs(n) for r in results for n in _numbers_in(r)} | {
        abs(n) for n in numbers(_DATE_TIME.sub(" ", question))
    }
    known |= {k * 100 for k in known if k <= 1} | {k / 100 for k in known}  # 0.367 = 36.7 %
    base = sorted(known)[:400]
    derived = set()
    for i, x in enumerate(base):
        for y in base[i + 1 :]:
            derived.add(y - x)
            if x >= 1:  # tiny denominators would make any large number look like a ratio
                pcts = {y / x * 100, (y - x) / x * 100, x / y * 100, (y - x) / y * 100}
                derived |= {p for p in pcts if p <= 1000}
    pool = known | derived

    def supported(v: float) -> bool:
        tol = max(0.011, abs(v) * 0.015)
        return any(abs(v - k) <= tol for k in pool)

    out = []
    for v in numbers(_DATE_TIME.sub(" ", answer)):
        a = abs(v)
        if a <= 31 and float(a).is_integer():
            continue  # counts, hours, list numbering
        if 1990 <= a <= 2100 and float(a).is_integer():
            continue  # years
        if not supported(a) and a not in out:
            out.append(a)
    return out


def _announced_tool(text: str, tools: dict[str, T.Tool]) -> str | None:
    """Name of a tool the model talks about instead of calling (qwen3 sometimes does)."""
    low = text.lower()
    return next((n for n in tools if n.lower() in low), None)


class Agent:
    def __init__(self, llm: Chat, tools: dict[str, T.Tool], max_steps: int = 6):
        self.llm = llm
        self.tools = tools
        self.max_steps = max_steps
        self.schemas = [t.schema() for t in tools.values()]
        self.history: list[dict] = [{"role": "system", "content": system_prompt(tools)}]

    def ask(self, question: str) -> Answer:
        t0 = time.perf_counter()
        self.history.append({"role": "user", "content": question})
        calls: list[dict] = []
        sources: list[str] = []
        nudged = grounded = False
        for step in range(1, self.max_steps + 1):
            reply = self.llm.chat(self.history, self.schemas)
            if not reply.tool_calls:
                self.history.append({"role": "assistant", "content": reply.content})
                # One push-back per question: an answer without any tool result is either
                # a refusal (kept, repeated unchanged) or an unsupported claim.
                if not nudged and not calls:
                    nudged = True
                    named = _announced_tool(reply.content, self.tools)
                    msg = NUDGE_ANNOUNCED.format(name=named) if named else NUDGE
                    self.history.append({"role": "user", "content": msg})
                    continue
                # Grounding: every figure must trace back to a tool result (or the question).
                results = [c["result"] for c in calls]
                loose = unsupported_numbers(reply.content, question, results)
                # One push-back per answer: only when tools were used (a tool-less answer
                # already got the "use a tool" push-back above); otherwise just report.
                if loose and calls and not grounded and step < self.max_steps:
                    grounded = True
                    shown = ", ".join(f"{n:g}" for n in loose[:8])
                    self.history.append(
                        {"role": "user", "content": NUDGE_GROUNDING.format(nums=shown)}
                    )
                    continue
                return Answer(
                    reply.content.strip(),
                    sources,
                    calls,
                    step,
                    time.perf_counter() - t0,
                    ungrounded=loose,
                )
            self.history.append(
                {
                    "role": "assistant",
                    "content": reply.content,
                    "tool_calls": [
                        {"function": {"name": c["name"], "arguments": c["arguments"]}}
                        for c in reply.tool_calls
                    ],
                }
            )
            for c in reply.tool_calls:
                result = T.call(self.tools, c["name"], c["arguments"])
                calls.append({"name": c["name"], "arguments": c["arguments"], "result": result})
                if "source" in result and result["source"] not in sources:
                    sources.append(result["source"])
                self.history.append(
                    {
                        "role": "tool",
                        "tool_name": c["name"],
                        "content": json.dumps(result, ensure_ascii=False, default=str),
                    }
                )
        # Out of tool rounds: ask for an answer without tools.
        self.history.append(
            {
                "role": "user",
                "content": "Answer now with the tool results you have; do not call more tools.",
            }
        )
        reply = self.llm.chat(self.history, None)
        self.history.append({"role": "assistant", "content": reply.content})
        return Answer(
            reply.content.strip(),
            sources,
            calls,
            self.max_steps,
            time.perf_counter() - t0,
            "max_steps",
        )

    def reset(self) -> None:
        self.history = self.history[:1]
