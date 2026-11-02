"""Evaluation set for the assistant: questions whose correct answer is computed from the data.

The truth is recomputed from the current gold/silver tables on every run (no hard-coded
figures), so the set stays valid as new weeks arrive. A case passes when the answer contains
the right number (within tolerance) or the right facts, and, for out-of-range questions,
when the assistant declines instead of inventing a figure.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import date, timedelta

import polars as pl

from ruhrwatt.rag import tools as T
from ruhrwatt.rag.agent import Answer

MONTHS_EN = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
]  # fmt: skip
MONTHS_DE = [
    "Januar", "Februar", "März", "April", "Mai", "Juni",
    "Juli", "August", "September", "Oktober", "November", "Dezember",
]  # fmt: skip
REFUSAL = (
    "not available", "no data", "not in the data", "only covers", "outside", "don't have",
    "do not have", "cannot", "can't", "nicht verfügbar", "keine daten", "liegen nicht",
    "nicht enthalten", "nur von", "not covered", "isn't available", "is not available",
)  # fmt: skip

from ruhrwatt.rag.numbers import numbers  # noqa: E402  (re-exported for callers)


def has_number(text: str, value: float, abs_tol: float = 0.0, rel_tol: float = 0.0) -> bool:
    tol = max(abs_tol, abs(value) * rel_tol)
    return any(abs(n - value) <= tol for n in numbers(text))


@dataclass
class Case:
    id: str
    question: str
    check: Callable[[Answer], tuple[bool, str]]
    expect_tools: tuple[str, ...] = ()


@dataclass
class Result:
    id: str
    passed: bool
    detail: str
    tools_used: list[str]
    seconds: float
    answer: str


def _num_case(
    cid: str, q: str, value: float, tools: tuple[str, ...], abs_tol=0.5, rel_tol=0.0
) -> Case:
    def check(a: Answer) -> tuple[bool, str]:
        ok = has_number(a.text, value, abs_tol, rel_tol)
        return ok, f"expected {value:g}" + ("" if ok else f"; numbers found: {numbers(a.text)[:6]}")

    return Case(cid, q, check, tools)


def _keyword_case(cid: str, q: str, any_of: tuple[str, ...], tools: tuple[str, ...]) -> Case:
    def check(a: Answer) -> tuple[bool, str]:
        low = a.text.lower()
        hit = [k for k in any_of if k.lower() in low]
        return bool(hit), f"looked for any of {list(any_of)}; found {hit}"

    return Case(cid, q, check, tools)


def build_cases() -> list[Case]:
    """Questions templated on the current data (latest complete months and days)."""
    mm = T.monthly_market()["months"]
    full = [m for m in mm if m["days"] >= 28]
    if len(full) < 2:
        raise ValueError("need at least two complete months of data for the eval set")
    last, prev = full[-1], full[-2]
    y, m = (int(x) for x in last["month"].split("-"))
    py, pm = (int(x) for x in prev["month"].split("-"))
    en, de = f"{MONTHS_EN[m - 1]} {y}", f"{MONTHS_DE[m - 1]} {y}"
    pen = f"{MONTHS_EN[pm - 1]} {py}"

    info = T.dataset_info()
    last_complete = date.fromisoformat(info["last_complete_quarter_hour_berlin"][:10])
    day = last_complete - timedelta(days=1)
    a, b = day - timedelta(days=13), day
    stats = T.price_stats(a.isoformat(), b.isoformat())
    block = T.cheapest_hours(day.isoformat(), 3, contiguous=True)
    custom = T.load_shift_savings(avg_mw=40, min_frac=0.5, max_frac=1.5)
    chlor = T.load_shift_savings(preset="chlor_alkali")
    chlor_narrow = T.load_shift_savings(preset="chlor_alkali", min_frac=0.9, max_frac=1.1)
    own = T.load_shift_savings(avg_mw=15, min_frac=0.8, max_frac=1.2)
    m_start = date(y, m, 1)
    m_end = (m_start + timedelta(days=32)).replace(day=1) - timedelta(days=1)
    weeks = T.period_breakdown(m_start.isoformat(), m_end.isoformat(), by="week", flat_mw=10)
    most_negative = T.monthly_market(sort_by="negative_price_hours", top=1)["months"][0]
    most_expensive = T.monthly_market(sort_by="price_base", top=1)["months"][0]
    # "the last 3 months" relative to today (or to the data end, if the data is older):
    # accept 3 complete calendar months or the last 91 days.
    ref = min(date.today(), last_complete + timedelta(days=1))
    cal_end = ref.replace(day=1) - timedelta(days=1)
    cal_start = (cal_end.replace(day=1) - timedelta(days=62)).replace(day=1)
    last3 = [
        T.load_shift_savings(
            avg_mw=25, min_frac=0.7, max_frac=1.2, start=a.isoformat(), end=b.isoformat()
        )
        for a, b in ((cal_start, cal_end), (ref - timedelta(days=91), ref - timedelta(days=1)))
    ]
    full25 = T.load_shift_savings(avg_mw=25, min_frac=0.7, max_frac=1.2)
    mid = m_start + timedelta(days=14)
    half1 = T.price_stats(m_start.isoformat(), mid.isoformat())
    half2 = T.price_stats((mid + timedelta(days=1)).isoformat(), m_end.isoformat())

    cases = [
        _num_case(
            "avg_price_month",
            f"What was the average day-ahead base price in {en}?",
            last["price_base"],
            ("monthly_market", "price_stats"),
        ),
        _num_case(
            "negative_hours_month",
            f"How many hours with negative prices were there in {pen}?",
            prev["negative_price_hours"],
            ("monthly_market", "price_stats"),
        ),
        _num_case(
            "peak_price_german",
            f"Wie hoch war der durchschnittliche Peak-Preis im {de}?",
            last["price_peak"],
            ("monthly_market", "price_stats"),
        ),
        Case(
            "renewables_de_vs_amprion",
            f"Compare the renewable share of Germany and the Amprion area in {en}.",
            lambda ans, last=last: (
                has_number(ans.text, last["de_renewables_share"], 0.6)
                and has_number(ans.text, last["amp_renewables_share"], 0.6),
                f"expected {last['de_renewables_share']} and {last['amp_renewables_share']} %",
            ),
            ("monthly_market",),
        ),
        Case(
            "max_price_period",
            f"What was the highest day-ahead price between {a} and {b}, and when was it?",
            lambda ans, s=stats: (
                has_number(ans.text, s["max_price"], 0.5)
                and _day_mentioned(ans.text, s["max_price_at_berlin"]),
                f"expected {s['max_price']} on {s['max_price_at_berlin']}",
            ),
            ("price_stats",),
        ),
        Case(
            "cheapest_block",
            f"On {day}, when was the cheapest 3-hour block to run a batch process?",
            lambda ans, blk=block: (
                blk["block"].split("-")[0] in ans.text,
                f"expected block {blk['block']}",
            ),
            ("cheapest_hours",),
        ),
        _num_case(
            "shift_custom",
            "How much could a 40 MW plant save per MWh if it can run between 50 % and 150 % "
            "of its average load?",
            custom["savings_eur_per_mwh"],
            ("load_shift_savings",),
            0.1,
        ),
        _num_case(
            "shift_preset_year",
            "What is the yearly savings potential of the chlor-alkali example plant?",
            chlor["savings_eur_per_year"],
            ("load_shift_savings",),
            rel_tol=0.02,
        ),
        # From the first live batch: a preset with the user's own limits must use the limits.
        _num_case(
            "shift_preset_override",
            "How much would the chlor-alkali example plant save per MWh if it could only flex "
            "between 90 % and 110 % of its load?",
            chlor_narrow["savings_eur_per_mwh"],
            ("load_shift_savings",),
            0.1,
        ),
        # ... and an own plant without a period must use the full history, not ask back.
        Case(
            "shift_own_plant_german",
            "Lohnt sich Lastverschiebung für uns? Wir haben 15 MW und können 20 % nach oben "
            "und unten flexen.",
            lambda ans, o=own: (
                has_number(ans.text, o["savings_eur_per_mwh"], 0.1)
                or has_number(ans.text, o["savings_eur_per_year"], rel_tol=0.02),
                f"expected {o['savings_eur_per_mwh']} EUR/MWh or {o['savings_eur_per_year']} EUR/a",
            ),
            ("load_shift_savings",),
        ),
        # ... and week rankings plus cost differences come from the tool, not from arithmetic.
        Case(
            "most_expensive_week",
            f"Which full week in {en} was the most expensive, and how much more would a "
            "constant 10 MW load have paid than in the cheapest full week?",
            lambda ans, w=weeks: (
                _day_mentioned(ans.text, w["most_expensive_period"][:10])
                and has_number(
                    ans.text, w["cost_difference_most_expensive_vs_cheapest_eur"], rel_tol=0.02
                ),
                f"expected week {w['most_expensive_period']} and "
                f"{w['cost_difference_most_expensive_vs_cheapest_eur']} EUR more",
            ),
            ("period_breakdown",),
        ),
        # From the second batch: rankings come from the tool, not from scanning a table ...
        Case(
            "most_negative_month",
            "Which complete month had the most hours with negative prices, and how many?",
            lambda ans, top=most_negative: (
                _month_mentioned(ans.text, top["month"])
                and has_number(ans.text, top["negative_price_hours"], 0.5),
                f"expected {top['month']} with {top['negative_price_hours']} h",
            ),
            ("monthly_market",),
        ),
        # ... and both halves of a comparison are fetched, none invented.
        Case(
            "month_halves",
            f"Compare the average day-ahead price in the first half of {en} "
            f"({m_start} to {mid}) with the second half.",
            lambda ans, h1=half1, h2=half2: (
                has_number(ans.text, h1["base_price_avg"], 0.5)
                and has_number(ans.text, h2["base_price_avg"], 0.5),
                f"expected {h1['base_price_avg']} and {h2['base_price_avg']}",
            ),
            ("price_stats",),
        ),
        # From the third batch: a relative period must reach the tool as start/end ...
        Case(
            "shift_last_3_months_german",
            "Wir haben eine Elektrolyse mit 25 MW, die zwischen 70 % und 120 % ihrer Last fahren "
            "kann. Wie viel hätte Lastverschiebung in den letzten 3 Monaten pro MWh gespart?",
            lambda ans, l3=last3, full=full25: (
                any(has_number(ans.text, x["savings_eur_per_mwh"], 0.1) for x in l3)
                and not has_number(ans.text, full["savings_eur"], rel_tol=0.01),
                f"expected {l3[0]['savings_eur_per_mwh']} ({l3[0]['period']}) or "
                f"{l3[1]['savings_eur_per_mwh']} ({l3[1]['period']}) EUR/MWh, "
                f"not the full-history {full['savings_eur_per_mwh']}",
            ),
            ("load_shift_savings",),
        ),
        # ... and a two-part ranking question ranks by price, then reads the gap.
        Case(
            "most_expensive_month_gap",
            "Which complete month was the most expensive (highest average day-ahead price), "
            "and how large was the gap between the peak and the off-peak price in that month?",
            lambda ans, top=most_expensive: (
                _month_mentioned(ans.text, top["month"])
                and any(  # "-19.5" or "19.5 lower" are both fine
                    has_number(ans.text, v, 0.5)
                    for v in (top["peak_minus_offpeak"], abs(top["peak_minus_offpeak"]))
                ),
                f"expected {top['month']} with a gap of {top['peak_minus_offpeak']} EUR/MWh",
            ),
            ("monthly_market",),
        ),
        _keyword_case(
            "doc_amprion",
            "Why does the project use Amprion data for NRW?",
            ("control area", "regelzone", "covers most of nrw", "nrw-level"),
            ("search_docs",),
        ),
        _keyword_case(
            "doc_conformal",
            "How is the P10-P90 price range calibrated?",
            ("conformal", "cqr", "konform"),
            ("search_docs",),
        ),
        Case(
            "out_of_range",
            "What was the average day-ahead price in 2019?",
            lambda ans: (
                any(k in ans.text.lower() for k in REFUSAL),
                "expected the assistant to say 2019 is not in the data",
            ),
            (),
        ),
    ]
    try:
        fq = T.forecast_quality()
        best = fq["models_by_mae"][0]
        cases.append(
            Case(
                "best_model",
                "Which price forecasting model performed best in the backtest, "
                "and what was its MAE?",
                lambda ans, best=best: (
                    best["model"].lower() in ans.text.lower()
                    and has_number(ans.text, best["mae"], 0.1),
                    f"expected {best['model']} with MAE {best['mae']}",
                ),
                ("forecast_quality",),
            )
        )
    except T.ToolError:
        pass  # no backtest yet
    return cases


def _month_mentioned(text: str, month: str) -> bool:
    y, m = (int(x) for x in month.split("-"))
    low = text.lower()
    forms = [month, f"{MONTHS_EN[m - 1]} {y}", f"{MONTHS_DE[m - 1]} {y}", f"{m:02d}/{y}"]
    return any(f.lower() in low for f in forms)


def _day_mentioned(text: str, ts: str) -> bool:
    d = date.fromisoformat(ts[:10])
    forms = [
        d.isoformat(), f"{d.day}.{d.month}.", f"{d.day:02d}.{d.month:02d}.",
        f"{d.day} {MONTHS_EN[d.month - 1]}", f"{MONTHS_EN[d.month - 1]} {d.day}",
        f"{d.day}. {MONTHS_DE[d.month - 1]}",
    ]  # fmt: skip
    return any(f.lower() in text.lower() for f in forms)


def run(
    make_agent: Callable[[], object],
    cases: list[Case] | None = None,
    progress: Callable[[Result], None] | None = None,
) -> list[Result]:
    """Fresh agent per case (no shared history), so cases are independent."""
    results = []
    for case in cases or build_cases():
        agent = make_agent()
        ans: Answer = agent.ask(case.question)
        ok, detail = case.check(ans)
        used = [c["name"] for c in ans.tool_calls]
        if case.expect_tools and not set(case.expect_tools) & set(used):
            ok, detail = (
                False,
                f"{detail}; expected a call to one of {list(case.expect_tools)}, got {used}",
            )
        r = Result(case.id, ok, detail, used, round(ans.seconds, 1), ans.text)
        results.append(r)
        if progress:
            progress(r)
    return results


def to_frame(results: list[Result], model: str) -> pl.DataFrame:
    return pl.DataFrame(
        [
            {
                "model": model,
                "case": r.id,
                "passed": r.passed,
                "detail": r.detail,
                "tools": ",".join(r.tools_used),
                "seconds": r.seconds,
                "answer": r.answer,
            }
            for r in results
        ]  # fmt: skip
    )
