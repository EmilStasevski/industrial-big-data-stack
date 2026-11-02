"""RAG assistant tests, offline: scripted LLM, hashing embedder, mocked Ollama HTTP."""

from __future__ import annotations

import hashlib
import json
import math
import random
import re
from datetime import datetime, timedelta
from pathlib import Path

import httpx
import polars as pl
import pytest

from ruhrwatt.rag import docs, evaluate
from ruhrwatt.rag import tools as T
from ruhrwatt.rag.agent import Agent
from ruhrwatt.rag.config import Settings
from ruhrwatt.rag.llm import ChatReply, OllamaClient, OllamaError
from ruhrwatt.silver import energy
from synthetic import _value, utc, write_bronze

ROOT = Path(__file__).resolve().parents[1]


# --------------------------------------------------------------------------- fixtures


def _price(ts: datetime) -> float:
    rng = random.Random(int(ts.timestamp()))
    return _value(energy.PRICE, ts) + 25 * math.sin(ts.timetuple().tm_yday / 3) + rng.gauss(0, 15)


@pytest.fixture(scope="module")
def data_dir(tmp_path_factory) -> Path:
    """Silver, gold and backtest outputs for 10 synthetic weeks (2 complete months)."""
    from ruhrwatt.gold import backtest, mart

    d = tmp_path_factory.mktemp("data")
    write_bronze(
        d / "bronze" / "smard", utc(2025, 9, 1), utc(2025, 11, 10), override={energy.PRICE: _price}
    )
    energy.build(d / "bronze" / "smard", d / "silver")
    out = mart.build(d / "silver", d / "gold")
    r = backtest.run(out["features"], test_weeks=2, models=("naive_d1", "ridge"))
    r["scores_df"].write_parquet(d / "gold" / "backtest_scores.parquet")
    r["predictions"].write_parquet(d / "gold" / "backtest_predictions.parquet")
    return d


@pytest.fixture
def data(data_dir: Path, monkeypatch) -> Path:
    monkeypatch.setenv("RUHRWATT_DATA_DIR", str(data_dir))
    return data_dir


def hash_embed(texts: list[str], dim: int = 256) -> list[list[float]]:
    """Deterministic bag-of-words embedding: shared words -> similar vectors."""
    out = []
    for t in texts:
        v = [0.0] * dim
        for w in re.findall(r"[a-zäöüß0-9]+", t.lower()):
            v[int(hashlib.md5(w.encode()).hexdigest(), 16) % dim] += 1.0
        n = math.sqrt(sum(x * x for x in v)) or 1.0
        out.append([x / n for x in v])
    return out


class Scripted:
    """LLM stand-in that returns prepared replies in order and records what it was sent."""

    def __init__(self, replies: list[ChatReply]):
        self.replies = list(replies)
        self.seen: list[list[dict]] = []

    def chat(self, messages, tools=None):
        self.seen.append([dict(m) for m in messages])
        return self.replies.pop(0)


# --------------------------------------------------------------------------- tools


def test_tools_return_rounded_numbers_and_sources(data) -> None:
    reg = T.registry()
    r = T.call(reg, "price_stats", {"start": "2025-09-01", "end": "2025-09-30"})
    assert r["days_with_prices"] == 30 and "source" in r
    daily = pl.read_parquet(data / "silver" / "energy_daily.parquet").filter(
        pl.col("date_berlin").dt.month() == 9
    )
    assert r["base_price_avg"] == round(daily["price_base"].mean(), 2)
    assert r["negative_price_hours_total"] == daily["price_negative_qh"].sum() / 4
    m = T.call(reg, "monthly_market", {"months": ["2025-10"]})["months"][0]
    assert m["month"] == "2025-10" and 0 < m["de_renewables_share"] < 100  # percent
    gold = pl.read_parquet(data / "gold" / "market_monthly.parquet").row(1, named=True)
    assert m["de_renewables_share"] == round(gold["de_renewables_share"] * 100, 2)


def test_cheapest_hours_individual_and_block(data) -> None:
    qh = pl.read_parquet(data / "silver" / "energy_15min.parquet").filter(
        pl.col("date_berlin") == datetime(2025, 10, 7).date()
    )
    hourly = qh.group_by(pl.col("ts_berlin").dt.hour().alias("h")).agg(pl.col(energy.PRICE).mean())
    cheapest = sorted(hourly.sort(energy.PRICE)["h"].head(2).to_list())
    r = T.cheapest_hours("2025-10-07", 2)
    assert [int(h["hour"][:2]) for h in r["hours"]] == cheapest
    b = T.cheapest_hours("2025-10-07", 3, contiguous=True)
    start, end = b["block"].split("-")
    assert (int(end[:2]) * 60 + int(end[3:])) - (int(start[:2]) * 60 + int(start[3:])) == 180
    assert b["block_avg_price"] <= r["avg_price_of_these_hours"] + 200  # sane magnitude


def test_tool_errors_go_back_to_the_model(data) -> None:
    reg = T.registry()
    assert (
        "no price data"
        in T.call(reg, "price_stats", {"start": "2019-01-01", "end": "2019-12-31"})["error"]
    )
    assert "must be a date" in T.call(reg, "price_stats", {"start": "march", "end": "x"})["error"]
    assert "not (yet) in the data" in T.call(reg, "cheapest_hours", {"day": "2030-01-01"})["error"]
    assert "unknown preset" in T.call(reg, "load_shift_savings", {"preset": "x"})["error"]
    assert "min_frac" in T.call(reg, "load_shift_savings", {"avg_mw": 10, "min_frac": 1.5})["error"]
    assert "unknown tool" in T.call(reg, "drop_tables", {})["error"]
    # unknown arguments are an error the model can fix, never silently dropped
    err = T.call(reg, "dataset_info", {"verbose": True})["error"]
    assert "does not accept ['verbose']" in err
    # ... but None values (unset optional arguments) are fine
    assert "first_day" in T.call(reg, "dataset_info", {"verbose": None})


def test_missing_tables_explain_what_to_run(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("RUHRWATT_DATA_DIR", str(tmp_path))
    assert "make pipeline" in T.call(T.registry(), "dataset_info", {})["error"]
    assert "make backtest" in T.call(T.registry(), "forecast_quality", {})["error"]


def test_tool_schemas_match_the_functions() -> None:
    import inspect

    for tool in T.registry(lambda query, k=4: {}).values():
        params = set(inspect.signature(tool.fn).parameters)
        declared = set(tool.parameters.get("properties", {}))
        assert declared <= params, tool.name
        assert set(tool.parameters.get("required", [])) <= declared
        assert tool.schema()["function"]["description"]


# --------------------------------------------------------------------------- docs index


def test_chunking_follows_headings_and_ignores_code() -> None:
    md = "# Title\nintro\n\n## A\ntext a\n```\n# not a heading\n```\n## B\n" + ("para\n\n" * 400)
    chunks = docs.chunk_markdown(md, "x.md", max_chars=300)
    heads = [c.heading for c in chunks]
    assert heads[0] == "Title" and heads[1] == "Title > A"
    assert "# not a heading" in chunks[1].text
    assert all(len(c.text) <= 300 for c in chunks if c.heading == "Title > B")
    assert sum(c.heading == "Title > B" for c in chunks) > 1


def test_index_reuses_unchanged_chunks_and_search_finds_the_right_doc(tmp_path) -> None:
    root = tmp_path / "proj"
    (root / "docs").mkdir(parents=True)
    (root / "README.md").write_text(
        "# Readme\nAmprion is the control area that covers most of NRW.\n"
    )
    (root / "docs" / "gold.md").write_text(
        "# Gold\n## Range\nThe price range is conformalized daily.\n"
    )
    calls: list[int] = []

    def embed(texts):
        calls.append(len(texts))
        return hash_embed(texts)

    idx = tmp_path / "idx"
    first = docs.build_index(embed, root, idx)
    assert first["embedded"] == first["chunks"] == 2
    again = docs.build_index(embed, root, idx)
    assert again["embedded"] == 0 and again["reused"] == 2
    (root / "docs" / "gold.md").write_text(
        "# Gold\n## Range\nCalibrated with conformal prediction.\n"
    )
    assert docs.build_index(embed, root, idx)["embedded"] == 1

    hits = docs.DocSearch(hash_embed, idx).search("which control area covers NRW", k=1)
    assert hits["results"][0]["source"] == "README.md > Readme"
    assert "README.md" in hits["source"]


def test_real_project_docs_are_indexable() -> None:
    chunks = docs.load_chunks(ROOT)
    sources = {c.source for c in chunks}
    assert {"README.md", "gold_schema.md", "silver_schema.md", "spark.md"} <= sources
    assert any("conformal" in c.text.lower() for c in chunks)


def test_search_without_index_returns_a_hint(tmp_path) -> None:
    r = docs.DocSearch(hash_embed, tmp_path / "none").search("x")
    assert "make rag-index" in r["error"]


# --------------------------------------------------------------------------- agent


def test_agent_runs_tools_and_cites_sources(data) -> None:
    llm = Scripted(
        [
            ChatReply("", [{"name": "monthly_market", "arguments": {"months": ["2025-09"]}}]),
            ChatReply("The base price in September 2025 was X EUR/MWh."),
        ]
    )
    agent = Agent(llm, T.registry())
    ans = agent.ask("What was the price in September 2025?")
    assert ans.text.startswith("The base price") and ans.steps == 2
    assert ans.sources == ["gold/market_monthly"]
    assert ans.tool_calls[0]["result"]["months"][0]["month"] == "2025-09"
    sent = llm.seen[1]
    assert sent[0]["role"] == "system" and "Data covers 2025-09-01" in sent[0]["content"]
    tool_msg = sent[-1]
    assert tool_msg["role"] == "tool" and tool_msg["tool_name"] == "monthly_market"
    assert json.loads(tool_msg["content"])["source"] == "gold/market_monthly"


def test_agent_stops_after_max_steps(data) -> None:
    loop = [ChatReply("", [{"name": "dataset_info", "arguments": {}}])] * 3
    llm = Scripted([*loop, ChatReply("Best effort answer.")])
    ans = Agent(llm, T.registry(), max_steps=3).ask("q")
    assert ans.stopped == "max_steps" and ans.text == "Best effort answer."
    assert "do not call more tools" in llm.seen[-1][-1]["content"]


def test_chat_keeps_context_and_reset_clears_it(data) -> None:
    from ruhrwatt.rag.agent import NUDGE

    llm = Scripted([ChatReply("a"), ChatReply("a"), ChatReply("b"), ChatReply("b")])
    agent = Agent(llm, T.registry())
    agent.ask("first")
    agent.ask("second")
    users = [m["content"] for m in llm.seen[-1] if m["role"] == "user" and m["content"] != NUDGE]
    assert users == ["first", "second"]
    agent.reset()
    assert len(agent.history) == 1


def test_answer_without_tools_gets_one_nudge(data) -> None:
    from ruhrwatt.rag.agent import NUDGE

    llm = Scripted(
        [
            ChatReply("Amprion is the TSO for NRW."),  # no tool: pushed back once
            ChatReply("", [{"name": "dataset_info", "arguments": {}}]),
            ChatReply("Answer with data."),
        ]
    )
    ans = Agent(llm, T.registry()).ask("Why Amprion?")
    assert ans.text == "Answer with data."
    assert [c["name"] for c in ans.tool_calls] == ["dataset_info"]
    assert llm.seen[1][-1]["content"] == NUDGE


def test_refusal_is_kept_after_the_nudge(data) -> None:
    no = ChatReply("No gas prices in the data.")
    ans = Agent(Scripted([no, no]), T.registry()).ask("Gas price?")
    assert ans.text == "No gas prices in the data." and ans.steps == 2 and not ans.tool_calls


def test_announced_but_not_made_tool_call_is_requested(data) -> None:
    llm = Scripted(
        [
            ChatReply("Um die Antwort zu erhalten, rufe ich forecast_quality auf."),
            ChatReply("", [{"name": "forecast_quality", "arguments": {}}]),
            ChatReply("Ridge ist am besten."),
        ]
    )
    ans = Agent(llm, T.registry()).ask("Wie gut ist die Prognose?")
    assert "Call forecast_quality now" in llm.seen[1][-1]["content"]
    assert ans.tool_calls[0]["name"] == "forecast_quality"


# ------------------------------------------------ fixes from the first live batch


def test_preset_values_are_overridden_by_explicit_arguments(data) -> None:
    base = T.load_shift_savings(preset="chlor_alkali")
    narrow = T.load_shift_savings(preset="chlor_alkali", min_frac=0.9, max_frac=1.1)
    assert narrow["avg_mw"] == base["avg_mw"] == 60
    assert narrow["flexibility"] == "90%-110% of average load"
    assert narrow["savings_eur_per_mwh"] < base["savings_eur_per_mwh"]
    assert "modified" in narrow["plant"] and narrow["plant_de"] == "Chlor-Alkali-Elektrolyse"
    own = T.load_shift_savings(preset="generic_10pct", avg_mw=25, min_frac=0.7, max_frac=1.2)
    assert own["avg_mw"] == 25 and own["min_load_mw"] == 17.5 and own["max_load_mw"] == 30


def test_short_period_savings_carry_a_warning(data) -> None:
    one_day = T.load_shift_savings(preset="aluminium", start="2025-10-07", end="2025-10-07")
    assert one_day["days"] == 1 and "extrapolated" in one_day["warning"]
    assert "warning" not in T.load_shift_savings(preset="aluminium")


def test_flat_load_cost_is_computed_by_the_tool(data) -> None:
    r = T.price_stats("2025-10-06", "2025-10-12", flat_mw=10)
    a, b = datetime(2025, 10, 6).date(), datetime(2025, 10, 12).date()
    qh = pl.read_parquet(data / "silver" / "energy_15min.parquet").filter(
        pl.col("date_berlin").is_between(a, b)
    )
    assert r["flat_load_cost_eur"] == pytest.approx(qh[energy.PRICE].sum() * 2.5, abs=0.01)
    assert r["flat_load_energy_mwh"] == 7 * 24 * 10


def test_period_breakdown_ranks_complete_weeks(data) -> None:
    r = T.period_breakdown("2025-09-29", "2025-11-09", by="week", flat_mw=10)
    weeks = r["periods"]
    assert weeks[0]["period"] == "2025-09-29 .. 2025-10-05" and all(w["days"] == 7 for w in weeks)
    dear = max(weeks, key=lambda w: w["avg_price"])
    cheap = min(weeks, key=lambda w: w["avg_price"])
    assert r["most_expensive_period"] == dear["period"] and r["cheapest_period"] == cheap["period"]
    assert r["cost_difference_most_expensive_vs_cheapest_eur"] == pytest.approx(
        dear["flat_load_cost_eur"] - cheap["flat_load_cost_eur"], abs=0.02
    )
    october = T.period_breakdown("2025-10-01", "2025-10-31", by="week")
    ranked = {october["cheapest_period"], october["most_expensive_period"]}
    assert all(p["days"] == 7 for p in october["periods"] if p["period"] in ranked)
    assert len(T.period_breakdown("2025-10-01", "2025-10-03", by="day")["periods"]) == 3


def test_monthly_rows_flag_partial_months(data) -> None:
    months = T.monthly_market()["months"]
    assert all(m["complete_month"] == (m["days"] >= 28) for m in months)


# ------------------------------------------------ fixes from the second live batch


def test_monthly_ranking_returns_rank_one_first(data) -> None:
    all_months = [m for m in T.monthly_market()["months"] if m["complete_month"]]
    best = max(all_months, key=lambda m: m["negative_price_hours"])
    r = T.monthly_market(sort_by="negative_price_hours", top=2)
    assert r["months"][0]["rank"] == 1 and r["months"][0]["month"] == best["month"]
    assert len(r["months"]) == 2 and "rank 1" in r["ranking"]
    low = T.monthly_market(sort_by="price_base", order="lowest", top=1)["months"][0]
    assert low["price_base"] == min(m["price_base"] for m in all_months)
    one_year = T.monthly_market(year=2025)["months"]
    assert {m["month"][:4] for m in one_year} == {"2025"}
    assert "peak_minus_offpeak" in one_year[0]
    assert "sort_by must be" in T.call(T.registry(), "monthly_market", {"sort_by": "x"})["error"]


def test_dropped_filter_cannot_change_the_answer(data) -> None:
    """Second live batch: start/end sent to monthly_market were dropped, and the ranking
    over all months answered a question about September."""
    reg = T.registry()
    bad = T.call(
        reg,
        "monthly_market",
        {"start": "2025-09-01", "end": "2025-09-30", "sort_by": "price_base", "top": 1},
    )
    assert "does not accept ['end', 'start']" in bad["error"] and "months" in bad["error"]
    good = T.call(reg, "monthly_market", {"months": ["2025-09"]})
    assert good["months"][0]["month"] == "2025-09"
    assert (
        "order must be"
        in T.call(reg, "monthly_market", {"sort_by": "price_base", "order": "max"})["error"]
    )


def test_requests_beyond_the_data_carry_a_warning(data) -> None:
    r = T.price_stats("2023-01-01", "2025-10-31")
    assert "prices exist only for 2025-09-01" in r["warning"]
    assert "warning" not in T.price_stats("2025-09-01", "2025-09-30")
    assert "warning" in T.period_breakdown("2025-10-01", "2026-12-31")
    assert "warning" in T.load_shift_savings(preset="aluminium", start="2020-01-01")


def test_forecast_quality_lists_the_real_features(data) -> None:
    imp = pl.DataFrame({"feature": ["price_lag_1d", "de_fc_residual_proxy"], "share": [0.6, 0.4]})
    imp.write_parquet(data / "gold" / "feature_importance.parquet")
    try:
        r = T.forecast_quality()
        assert r["top_features_lightgbm"][0] == {
            "feature": "price_lag_1d",
            "share_of_gain_pct": 60.0,
        }
        assert "no weather" in r["feature_notes"]
    finally:
        (data / "gold" / "feature_importance.parquet").unlink()


def test_grounding_flags_invented_numbers_only() -> None:
    from ruhrwatt.rag.agent import unsupported_numbers

    res = [{"base_price_avg": 159.89, "max_price": 406.0, "skill_vs_naive_d1": 0.367}]
    assert unsupported_numbers("Die Hälften: 159,89 und 178,46 EUR/MWh", "Oktober", res) == [178.46]
    ok = "159,89 EUR/MWh, max 406 EUR/MWh (2026-10-14 18:00), 37 % besser, 246,11 Abstand"
    assert unsupported_numbers(ok, "", res) == []  # rounding, percent, difference, dates
    assert unsupported_numbers("10 MW über 3 Monate", "10 MW, 3 Monate", res) == []


def test_agent_pushes_back_on_ungrounded_numbers(data) -> None:
    from ruhrwatt.rag.agent import NUDGE_GROUNDING

    llm = Scripted(
        [
            ChatReply(
                "",
                [
                    {
                        "name": "price_stats",
                        "arguments": {"start": "2025-10-01", "end": "2025-10-15"},
                    }
                ],
            ),
            ChatReply("Erste Hälfte X, zweite Hälfte 9.876,54 EUR/MWh."),  # invented
            ChatReply("Für die zweite Hälfte habe ich keine Zahl abgefragt."),
        ]
    )
    ans = Agent(llm, T.registry()).ask("Vergleiche die Oktoberhälften 2025.")
    assert "9876.54" in llm.seen[2][-1]["content"]
    assert llm.seen[2][-1]["content"].startswith(NUDGE_GROUNDING.split("{")[0])
    assert ans.text.startswith("Für die zweite Hälfte") and ans.ungrounded == []


def test_ungrounded_numbers_are_reported_when_the_model_insists(data) -> None:
    call = ChatReply("", [{"name": "dataset_info", "arguments": {}}])
    claim = ChatReply("Der Preis lag bei 123,45 EUR/MWh.")
    ans = Agent(Scripted([call, claim, claim]), T.registry()).ask("Preis?")
    assert ans.ungrounded == [123.45]
    from ruhrwatt.rag import batch

    row = {
        "id": "x",
        "category": "c",
        "question": "Preis?",
        "answer": ans.text,
        "tools": [],
        "tool_errors": [],
        "seconds": 1.0,
        "stopped": "answered",
        "ungrounded": ans.ungrounded,
    }
    assert "not found in any tool result: 123.45" in batch.to_markdown([row], "fake")


# --------------------------------------------------------------------------- Ollama client


def _client(handler, model="qwen3:8b") -> OllamaClient:
    from dataclasses import replace

    return OllamaClient(replace(Settings(), model=model), transport=httpx.MockTransport(handler))


def test_chat_request_and_tool_call_parsing() -> None:
    sent = {}

    def handler(req: httpx.Request) -> httpx.Response:
        sent.update(json.loads(req.content))
        return httpx.Response(
            200,
            json={
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {"function": {"name": "price_stats", "arguments": {"start": "2026-01-01"}}},
                        {"function": {"name": "dataset_info", "arguments": "{}"}},
                    ],
                },
                "eval_count": 12,
                "total_duration": 2_000_000_000,
            },
        )

    r = _client(handler).chat([{"role": "user", "content": "hi"}], [{"type": "function"}])
    assert sent["model"] == "qwen3:8b" and sent["stream"] is False and sent["think"] is False
    assert sent["options"]["temperature"] == 0 and sent["tools"] == [{"type": "function"}]
    assert r.tool_calls == [
        {"name": "price_stats", "arguments": {"start": "2026-01-01"}},
        {"name": "dataset_info", "arguments": {}},
    ]
    assert r.seconds == 2.0 and r.eval_tokens == 12


def test_think_flag_only_sent_for_qwen3_by_default() -> None:
    sent = {}

    def handler(req):
        sent.update(json.loads(req.content))
        return httpx.Response(200, json={"message": {"content": "ok"}})

    _client(handler, model="llama3.1:8b").chat([])
    assert "think" not in sent


def test_missing_model_and_unreachable_server_give_clear_errors() -> None:
    def missing(req):
        return httpx.Response(404, json={"error": "model 'qwen3:8b' not found"})

    with pytest.raises(OllamaError, match="ollama pull qwen3:8b"):
        _client(missing).chat([])

    def down(req):
        raise httpx.ConnectError("refused")

    with pytest.raises(OllamaError, match="ollama serve"):
        _client(down).chat([])


def test_embed_parses_vectors() -> None:
    def handler(req):
        body = json.loads(req.content)
        assert body["model"] == "bge-m3"
        return httpx.Response(200, json={"embeddings": [[0.1, 0.2]] * len(body["input"])})

    assert _client(handler).embed(["a", "b"]) == [[0.1, 0.2], [0.1, 0.2]]


# --------------------------------------------------------------------------- evaluation


@pytest.mark.parametrize(
    ("text", "value", "ok"),
    [
        ("The average was 99,72 EUR/MWh.", 99.72, True),
        ("about 3.82 million EUR per year", 3_819_818, False),
        ("about 3.82 million EUR per year", 3_820_000, True),
        ("3,819,818 EUR", 3_819_818, True),
        ("3.819.818 EUR", 3_819_818, True),
        ("1.234,5 MWh", 1234.5, True),
        ("minus −12.5", -12.5, True),
        ("4 Mio. EUR", 4_000_000, True),
        ("costs 120 MWh", 120, True),
        ("no numbers here", 1.0, False),
    ],
)
def test_number_extraction(text: str, value: float, ok: bool) -> None:
    assert evaluate.has_number(text, value, abs_tol=0.01) is ok


MONTHS = {m.lower(): i + 1 for i, m in enumerate(evaluate.MONTHS_EN)} | {
    m.lower(): i + 1 for i, m in enumerate(evaluate.MONTHS_DE)
}


class FakeAnalyst:
    """Rule-based stand-in for the LLM: picks a tool from the question, then reports the
    tool result. Exercises the whole eval harness (questions, tools, checks) offline."""

    def __init__(self, searcher):
        self.searcher = searcher

    def chat(self, messages, tools=None):
        last = messages[-1]
        if last["role"] == "tool":
            turn = []
            for m in reversed(messages):
                if m["role"] == "user":
                    break
                if m["role"] == "tool":
                    turn.append(json.loads(m["content"]))
            if any("error" in r for r in turn):
                errors = "; ".join(r["error"] for r in turn if "error" in r)
                return ChatReply(f"That is not available in the data: {errors}")
            return ChatReply("Here is what I found: " + json.dumps(turn, ensure_ascii=False))
        q = last["content"]
        ql = q.lower()
        dates = re.findall(r"\d{4}-\d{2}-\d{2}", q)
        month = next(
            (
                f"{y}-{MONTHS[n.lower()]:02d}"
                for n, y in re.findall(r"([A-Za-zä]+) (\d{4})", q)
                if n.lower() in MONTHS
            ),
            None,
        )
        calls = None
        if "letzten 3 monaten" in ql:
            last = datetime.fromisoformat(
                T.dataset_info()["last_complete_quarter_hour_berlin"][:10]
            )
            ref = min(datetime.now(), last + timedelta(days=1)).date()
            end = ref.replace(day=1) - timedelta(days=1)
            start = (end.replace(day=1) - timedelta(days=62)).replace(day=1)
            args = {"avg_mw": 25, "min_frac": 0.7, "max_frac": 1.2}
            calls = [("load_shift_savings", args | {"start": str(start), "end": str(end)})]
        elif "most expensive (highest average" in ql:
            calls = [("monthly_market", {"sort_by": "price_base", "top": 1})]
        elif "most hours with negative" in ql:
            calls = [("monthly_market", {"sort_by": "negative_price_hours", "top": 1})]
        elif "first half" in ql:
            d1, d2 = dates[0], dates[1]
            nxt = (datetime.fromisoformat(d2) + timedelta(days=1)).date()
            y, mo = (int(x) for x in month.split("-"))
            last_day = (datetime(y + mo // 12, mo % 12 + 1, 1) - timedelta(days=1)).date()
            calls = [
                ("price_stats", {"start": d1, "end": d2}),
                ("price_stats", {"start": str(nxt), "end": str(last_day)}),
            ]
        if calls:
            return ChatReply("", [{"name": n, "arguments": a} for n, a in calls])
        if "2019" in q:
            call = ("price_stats", {"start": "2019-01-01", "end": "2019-12-31"})
        elif "15 mw" in ql:
            call = ("load_shift_savings", {"avg_mw": 15, "min_frac": 0.8, "max_frac": 1.2})
        elif "chlor" in ql and "90 %" in ql:
            args = {"preset": "chlor_alkali", "min_frac": 0.9, "max_frac": 1.1}
            call = ("load_shift_savings", args)
        elif "full week" in ql:
            y, mo = (int(x) for x in month.split("-"))
            first = datetime(y, mo, 1).date()
            last = (datetime(y + mo // 12, mo % 12 + 1, 1) - timedelta(days=1)).date()
            args = {"start": str(first), "end": str(last), "by": "week", "flat_mw": 10}
            call = ("period_breakdown", args)
        elif "40 mw" in ql:
            call = ("load_shift_savings", {"avg_mw": 40, "min_frac": 0.5, "max_frac": 1.5})
        elif "between" in ql:
            call = ("price_stats", {"start": dates[0], "end": dates[1]})
        elif "block" in ql:
            call = ("cheapest_hours", {"day": dates[0], "hours": 3, "contiguous": True})
        elif "chlor" in ql:
            call = ("load_shift_savings", {"preset": "chlor_alkali"})
        elif "model" in ql:
            call = ("forecast_quality", {})
        elif ql.startswith(("why", "how is")):
            call = ("search_docs", {"query": q, "k": 8})  # crude test embedder: wider net
        else:
            call = ("monthly_market", {"months": [month]})
        return ChatReply("", [{"name": call[0], "arguments": call[1]}])


def test_eval_harness_end_to_end(data, tmp_path) -> None:
    idx = tmp_path / "idx"
    docs.build_index(hash_embed, ROOT, idx)
    searcher = docs.DocSearch(hash_embed, idx)
    reg = T.registry(docs.search_docs_tool(searcher))
    cases = evaluate.build_cases()
    assert {c.id for c in cases} >= {
        "avg_price_month",
        "out_of_range",
        "doc_conformal",
        "best_model",
    }
    results = evaluate.run(lambda: Agent(FakeAnalyst(searcher), reg), cases)
    failed = {r.id: r.detail for r in results if not r.passed}
    assert not failed, failed
    frame = evaluate.to_frame(results, "fake")
    assert frame.height == len(cases) and frame["passed"].all()


def test_eval_fails_a_made_up_number(data) -> None:
    case = next(c for c in evaluate.build_cases() if c.id == "avg_price_month")
    llm = Scripted(
        [
            ChatReply("", [{"name": "monthly_market", "arguments": {}}]),
            ChatReply("It was 12.34 EUR/MWh."),
        ]
    )
    [r] = evaluate.run(lambda: Agent(llm, T.registry()), [case])
    assert not r.passed and "expected" in r.detail


def test_eval_requires_the_expected_tool(data) -> None:
    """The right number without a tool call is a guess, and fails."""
    case = next(c for c in evaluate.build_cases() if c.id == "avg_price_month")
    latest = [m for m in T.monthly_market()["months"] if m["days"] >= 28][-1]
    guess = ChatReply(f"Probably {latest['price_base']} EUR/MWh.")
    llm = Scripted([guess, guess])  # no tool call, even after the push-back
    [r] = evaluate.run(lambda: Agent(llm, T.registry()), [case])
    assert not r.passed and "expected a call" in r.detail
    assert evaluate.has_number(r.answer, latest["price_base"])  # number right, still fails


# --------------------------------------------------------------------------- batch


def test_german_question_file_is_valid() -> None:
    from ruhrwatt.rag import batch

    items = batch.load_questions(ROOT / "questions" / "fragen_de.json")
    assert len(items) >= 20
    assert any(it.get("conversation") for it in items)


def test_batch_runs_conversations_with_shared_context(data, tmp_path) -> None:
    from ruhrwatt.rag import batch

    items = [
        {"id": "a", "category": "x", "question": "eins"},
        {"id": "b", "category": "y", "conversation": ["zwei", "drei"]},
    ]
    agents = []

    def make():
        llm = Scripted(
            [
                ChatReply(
                    "",
                    [
                        {
                            "name": "price_stats",
                            "arguments": {"start": "2019-01-01", "end": "2019-01-02"},
                        }
                    ],
                ),
                ChatReply("Keine Daten."),
                ChatReply("Antwort 2"),
                ChatReply("Antwort 2"),  # repeated after the push-back
            ]
        )
        agents.append(llm)
        return Agent(llm, T.registry())

    res = batch.run(make, items)
    assert [r["id"] for r in res] == ["a", "b.1", "b.2"]
    assert (
        res[0]["tools"][0]["name"] == "price_stats" and "no price data" in res[0]["tool_errors"][0]
    )
    # the second turn of the conversation saw the first one
    from ruhrwatt.rag.agent import NUDGE

    seen = agents[1].seen[-1]
    users = [m["content"] for m in seen if m["role"] == "user" and m["content"] != NUDGE]
    assert users == ["zwei", "drei"]
    md = batch.to_markdown(res, "fake")
    assert "## b.2 (y)" in md and "price_stats" in md and "tool errors" in md


def test_batch_keeps_going_after_an_error_but_stops_on_fatal(data) -> None:
    from ruhrwatt.rag import batch

    class Boom:
        def ask(self, q):
            raise RuntimeError("model crashed")

    res = batch.run(lambda: Boom(), [{"id": "a", "question": "x"}, {"id": "b", "question": "y"}])
    assert [r["error"] for r in res] == ["RuntimeError: model crashed"] * 2
    with pytest.raises(OllamaError):

        class Down:
            def ask(self, q):
                raise OllamaError("down")

        batch.run(lambda: Down(), [{"id": "a", "question": "x"}], fatal=(OllamaError,))


def test_bad_question_file_is_rejected(tmp_path) -> None:
    from ruhrwatt.rag import batch

    p = tmp_path / "q.json"
    p.write_text('{"questions": [{"id": "a"}]}')
    with pytest.raises(ValueError, match="needs 'question'"):
        batch.load_questions(p)
    p.write_text('{"questions": [{"id": "a", "question": "x"}, {"id": "a", "question": "y"}]}')
    with pytest.raises(ValueError, match="duplicate"):
        batch.load_questions(p)


def test_full_history_sold_as_last_3_months_fails_the_eval(data) -> None:
    """Third live batch: the model dropped 'last 3 months' and reported full-history savings."""
    case = next(c for c in evaluate.build_cases() if c.id == "shift_last_3_months_german")
    full = T.load_shift_savings(avg_mw=25, min_frac=0.7, max_frac=1.2)
    call = ChatReply(
        "",
        [
            {
                "name": "load_shift_savings",
                "arguments": {"avg_mw": 25, "min_frac": 0.7, "max_frac": 1.2},
            }
        ],
    )
    text = ChatReply(
        f"In den letzten 3 Monaten: {full['savings_eur']} EUR, "
        f"{full['savings_eur_per_mwh']} EUR/MWh."
    )
    [r] = evaluate.run(lambda: Agent(Scripted([call, text, text]), T.registry()), [case])
    assert not r.passed and "full-history" in r.detail
