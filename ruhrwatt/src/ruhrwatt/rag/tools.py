"""Tools the assistant calls for every number it states.

Each tool validates its arguments, reads only what it needs (Polars lazy scans with
predicate and projection pushdown), rounds results, and returns a ``source`` string that the
answer cites. Errors come back as ``{"error": ...}`` so the model can correct itself.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import polars as pl

from ruhrwatt import config
from ruhrwatt.silver.energy import PRICE

ROUND = 2


class ToolError(ValueError):
    """Bad arguments or missing data; the message is shown to the model."""


def _silver(name: str) -> Path:
    p = config.data_dir() / "silver" / f"{name}.parquet"
    if not p.exists():
        raise ToolError(f"{p.name} is missing; run `make pipeline` first")
    return p


def _gold(name: str, hint: str = "make gold") -> Path:
    p = config.data_dir() / "gold" / f"{name}.parquet"
    if not p.exists():
        raise ToolError(f"{p.name} is missing; run `{hint}` first")
    return p


def _date(value: Any, name: str) -> date:
    try:
        return date.fromisoformat(str(value)[:10])
    except ValueError as exc:
        raise ToolError(f"{name} must be a date like 2026-03-01, got {value!r}") from exc


def _coverage(a: date, b: date) -> dict:
    """Warning when a requested range reaches beyond the days that have prices."""
    r = (
        pl.scan_parquet(_silver("energy_daily"))
        .filter(pl.col("price_base").is_not_null())
        .select(lo=pl.col("date_berlin").min(), hi=pl.col("date_berlin").max())
        .collect()
        .row(0, named=True)
    )
    if a >= r["lo"] and b <= r["hi"]:
        return {}
    lo, hi = max(a, r["lo"]), min(b, r["hi"])
    return {
        "warning": f"requested {a}..{b}, but prices exist only for {r['lo']}..{r['hi']}: "
        f"all figures refer to {lo}..{hi} only. Say so; do not describe the rest of the "
        "requested period."
    }


def _r(x: Any) -> Any:
    if isinstance(x, float):
        return round(x, ROUND)
    return x


def _row(d: dict) -> dict:
    return {k: (v.isoformat() if isinstance(v, date) else _r(v)) for k, v in d.items()}


# --------------------------------------------------------------------------- tools


def dataset_info() -> dict:
    """What data exists: date range, last complete quarter-hour, last published price."""
    qh = pl.scan_parquet(_silver("energy_15min"))
    r = (
        qh.select(
            first=pl.col("date_berlin").min(),
            last_complete=pl.col("ts_berlin").filter(pl.col("is_complete")).max(),
            last_price=pl.col("ts_berlin").filter(pl.col(PRICE).is_not_null()).max(),
        )
        .collect()
        .row(0, named=True)
    )
    return {
        "first_day": r["first"].isoformat(),
        "last_complete_quarter_hour_berlin": f"{r['last_complete']:%Y-%m-%d %H:%M}",
        "last_published_price_berlin": f"{r['last_price']:%Y-%m-%d %H:%M}",
        "bidding_zone": "DE-LU (day-ahead)",
        "regions": "Germany (de_) and the Amprion control area, proxy for NRW (amp_)",
        "units": "prices EUR/MWh, energy MWh per quarter-hour",
        "source": "silver/energy_15min",
    }


def price_stats(start: str, end: str, flat_mw: float | None = None) -> dict:
    """Day-ahead price statistics for a date range (inclusive, Berlin days).

    ``flat_mw``: also return the day-ahead energy cost of a constant load of that size.
    """
    a, b = _date(start, "start"), _date(end, "end")
    if b < a:
        raise ToolError("end is before start")
    days = (
        pl.scan_parquet(_silver("energy_daily"))
        .filter(pl.col("date_berlin").is_between(a, b) & pl.col("price_base").is_not_null())
        .collect()
    )
    if days.height == 0:
        raise ToolError(f"no price data between {a} and {b}; call dataset_info for the range")
    qh = (
        pl.scan_parquet(_silver("energy_15min"))
        .filter(pl.col("date_berlin").is_between(a, b) & pl.col(PRICE).is_not_null())
        .select("ts_berlin", PRICE)
        .collect()
    )
    lo, hi = qh.sort(PRICE).row(0, named=True), qh.sort(PRICE).row(-1, named=True)
    w = days.drop_nulls("amp_load_weighted_price")
    out = {
        "period": f"{days['date_berlin'].min()} .. {days['date_berlin'].max()}",
        "days_with_prices": days.height,
        "base_price_avg": _r(days["price_base"].mean()),
        "peak_price_avg": _r(days["price_peak"].mean()),
        "offpeak_price_avg": _r(days["price_offpeak"].mean()),
        "min_price": _r(lo[PRICE]),
        "min_price_at_berlin": f"{lo['ts_berlin']:%Y-%m-%d %H:%M}",
        "max_price": _r(hi[PRICE]),
        "max_price_at_berlin": f"{hi['ts_berlin']:%Y-%m-%d %H:%M}",
        "negative_price_hours_total": _r(float(days["price_negative_qh"].sum()) / 4),
        "avg_daily_spread": _r(days["price_spread"].mean()),
        "amprion_load_weighted_price": _r(
            float(
                (w["amp_load_weighted_price"] * w["amp_load_mwh"]).sum() / w["amp_load_mwh"].sum()
            )
        )
        if w.height
        else None,
        "unit": "EUR/MWh",
        "source": f"silver/energy_daily + energy_15min, {a}..{b}",
    }
    if flat_mw:
        out |= _flat_cost(qh[PRICE], float(flat_mw))
    return out | _coverage(a, b)


def _flat_cost(prices: pl.Series, mw: float) -> dict:
    """Energy cost of a constant load: sum(price) * MW * 0.25 h."""
    return {
        "flat_load_mw": mw,
        "flat_load_energy_mwh": _r(len(prices) * mw * 0.25),
        "flat_load_cost_eur": _r(float(prices.sum()) * mw * 0.25),
    }


def period_breakdown(start: str, end: str, by: str = "week", flat_mw: float | None = None) -> dict:
    """Prices per week (Mon-Sun) or per day in a range, with the cheapest and most expensive
    period identified, and optionally the cost of a constant load in each period."""
    a, b = _date(start, "start"), _date(end, "end")
    if b < a:
        raise ToolError("end is before start")
    if by not in ("week", "day"):
        raise ToolError("by must be 'week' or 'day'")
    qh = (
        pl.scan_parquet(_silver("energy_15min"))
        .filter(pl.col("date_berlin").is_between(a, b) & pl.col(PRICE).is_not_null())
        .select("date_berlin", PRICE)
        .collect()
    )
    if qh.height == 0:
        raise ToolError(f"no price data between {a} and {b}; call dataset_info for the range")
    key = (
        pl.col("date_berlin").dt.truncate("1w") if by == "week" else pl.col("date_berlin")
    ).alias("period_start")
    agg = [
        pl.col("date_berlin").min().alias("first_day"),
        pl.col("date_berlin").max().alias("last_day"),
        pl.col("date_berlin").n_unique().alias("days"),
        pl.col(PRICE).mean().alias("avg_price"),
        pl.col(PRICE).min().alias("min_price"),
        pl.col(PRICE).max().alias("max_price"),
        ((pl.col(PRICE) < 0).sum() / 4).alias("negative_price_hours"),
    ]
    if flat_mw:
        agg.append((pl.col(PRICE).sum() * float(flat_mw) * 0.25).alias("flat_load_cost_eur"))
    per = qh.group_by(key).agg(agg).sort("period_start")
    skip = ("period_start", "first_day", "last_day", "days")
    rows = [
        {
            "period": f"{r['first_day']} .. {r['last_day']}",
            "days": r["days"],
            **{k: _r(r[k]) for k in r if k not in skip},
        }
        for r in per.iter_rows(named=True)
    ]
    full = [r for r in rows if by == "day" or r["days"] == 7] or rows
    cheapest = min(full, key=lambda r: r["avg_price"])
    dearest = max(full, key=lambda r: r["avg_price"])
    out = {
        "by": by,
        "periods": rows,
        "cheapest_period": cheapest["period"],
        "most_expensive_period": dearest["period"],
        "ranking_note": "weeks ranked only if complete (7 days)" if by == "week" else "",
        "unit": "EUR/MWh (time-weighted average of 15-minute prices)",
        "source": f"silver/energy_15min, {a}..{b}",
    }
    if flat_mw:
        out["cost_difference_most_expensive_vs_cheapest_eur"] = _r(
            dearest["flat_load_cost_eur"] - cheapest["flat_load_cost_eur"]
        )
    return out | _coverage(a, b)


MONTHLY_COLUMNS = (
    "month",
    "days",
    "price_base",
    "price_peak",
    "price_offpeak",
    "price_min",
    "price_max",
    "negative_price_hours",
    "price_daily_spread_avg",
    "de_renewables_share",
    "amp_renewables_share",
    "de_wind_pv_share",
    "amp_wind_pv_share",
    "amp_load_weighted_price",
)


RANKABLE = (
    "price_base",
    "price_peak",
    "price_offpeak",
    "peak_minus_offpeak",
    "price_min",
    "price_max",
    "negative_price_hours",
    "price_daily_spread_avg",
    "de_renewables_share",
    "amp_renewables_share",
    "de_wind_pv_share",
    "amp_wind_pv_share",
    "amp_load_weighted_price",
)


def monthly_market(
    months: list[str] | None = None,
    year: int | None = None,
    sort_by: str | None = None,
    order: str = "highest",
    top: int | None = None,
) -> dict:
    """Monthly market facts. ``months`` like ["2026-03"], ``year`` = one year, nothing = all.

    ``sort_by`` ranks complete months by a column (e.g. most negative-price hours), so the
    model never has to scan a long table itself.
    """
    df = pl.scan_parquet(_gold("market_monthly"))
    available = [c for c in MONTHLY_COLUMNS if c in df.collect_schema().names()]
    df = df.select(available).with_columns(
        peak_minus_offpeak=pl.col("price_peak") - pl.col("price_offpeak")
    )
    if months:
        wanted = []
        for m in months[:24]:
            d = _date(f"{str(m)[:7]}-01", "month")
            wanted.append(d)
        df = df.filter(pl.col("month").is_in(wanted))
    if year:
        df = df.filter(pl.col("month").dt.year() == int(year))
    ranking = None
    if sort_by:
        if sort_by not in RANKABLE:
            raise ToolError(f"sort_by must be one of {list(RANKABLE)}")
        if order not in ("highest", "lowest"):
            raise ToolError("order must be 'highest' or 'lowest'")
        df = df.filter(pl.col("days") >= 28).sort(sort_by, descending=order == "highest")
        df = df.head(int(top or 3))
        ranking = f"complete months ranked by {sort_by}, {order} first; rank 1 is the answer"
    else:
        df = df.sort("month")
    rows = df.collect()
    if rows.height == 0:
        raise ToolError(f"no monthly data for {months or year}; call dataset_info for the range")
    out = []
    for n, r in enumerate(rows.iter_rows(named=True), 1):
        for k in [k for k in r if k.endswith("_share")]:
            r[k] = r[k] * 100 if r[k] is not None else None  # percent, before rounding
        r = _row(r)
        r["month"] = r["month"][:7]
        r["complete_month"] = r["days"] >= 28
        if ranking:
            r = {"rank": n, **r}
        out.append(r)
    if ranking:
        return {
            "ranking": ranking,
            "months": out,
            "notes": "prices EUR/MWh; *_share in % of load",
            "source": "gold/market_monthly",
        }
    return {
        "months": out,
        "notes": (
            "prices EUR/MWh; *_share in % of load; negative_price_hours = total hours below 0 "
            "in the month; complete_month=false for partial months"
        ),
        "source": "gold/market_monthly",
    }


def cheapest_hours(
    day: str, hours: int = 4, contiguous: bool = False, most_expensive: bool = False
) -> dict:
    """Cheapest (or most expensive) hours of one Berlin day, from the day-ahead prices.

    ``contiguous`` = one block of ``hours`` in a row (e.g. a batch process), otherwise the
    individually cheapest hours.
    """
    d = _date(day, "day")
    n = int(hours)
    if not 1 <= n <= 23:
        raise ToolError("hours must be between 1 and 23")
    qh = (
        pl.scan_parquet(_silver("energy_15min"))
        .filter((pl.col("date_berlin") == d) & pl.col(PRICE).is_not_null())
        .select("ts_berlin", PRICE)
        .sort("ts_berlin")
        .collect()
    )
    if qh.height < 88:
        raise ToolError(
            f"day-ahead prices for {d} are not (yet) in the data. The auction closes at 12:00 "
            "the day before, but SMARD shows the results only later in the afternoon or "
            "evening; the pipeline runs at 14:15 and 19:30 Berlin time"
        )
    label = "most expensive" if most_expensive else "cheapest"
    if contiguous:
        k = 4 * n
        roll = qh.with_columns(avg=pl.col(PRICE).rolling_mean(k)).drop_nulls("avg")
        best = roll.sort("avg", descending=most_expensive).row(0, named=True)
        end = best["ts_berlin"] + timedelta(minutes=15)
        start = end - timedelta(hours=n)
        result = {
            "block": f"{start:%H:%M}-{end:%H:%M}",
            "block_avg_price": _r(best["avg"]),
        }
    else:
        hourly = (
            qh.group_by(pl.col("ts_berlin").dt.truncate("1h").alias("hour"))
            .agg(pl.col(PRICE).mean().alias("price"))
            .sort("price", descending=most_expensive)
            .head(n)
            .sort("hour")
        )
        result = {
            "hours": [
                {
                    "hour": f"{r['hour']:%H:%M}-{r['hour'] + timedelta(hours=1):%H:%M}",
                    "price": _r(r["price"]),
                }
                for r in hourly.iter_rows(named=True)
            ],
            "avg_price_of_these_hours": _r(hourly["price"].mean()),
        }
    return {
        "day": d.isoformat(),
        "selection": f"{label} {n} h" + (" in a row" if contiguous else ""),
        **result,
        "day_avg_price": _r(qh[PRICE].mean()),
        "unit": "EUR/MWh, hourly averages of 15-minute prices",
        "source": f"silver/energy_15min, {d}",
    }


PRESET_DE = {
    "generic_10pct": "Referenzanlage mit +/-10 % Flexibilität",
    "chlor_alkali": "Chlor-Alkali-Elektrolyse",
    "aluminium": "Aluminiumhütte (Elektrolyseöfen)",
    "steel_eaf": "Elektrolichtbogenofen (Stahl)",
}


def load_shift_savings(
    avg_mw: float | None = None,
    min_frac: float | None = None,
    max_frac: float | None = None,
    preset: str | None = None,
    start: str | None = None,
    end: str | None = None,
) -> dict:
    """Savings from shifting a plant's load within each day to cheaper quarter-hours.

    A preset gives starting values; avg_mw / min_frac / max_frac given explicitly override
    them (e.g. "the chlor-alkali plant, but only 90-110 %"). No start/end = full history.
    """
    from ruhrwatt.gold import shift

    base = None
    if preset:
        if preset not in shift.PRESETS:
            raise ToolError(f"unknown preset {preset!r}; choose from {list(shift.PRESETS)}")
        base = shift.PRESETS[preset]
    if base is None and avg_mw is None:
        raise ToolError("give avg_mw (plant average load in MW) or a preset")
    given = (("avg_mw", avg_mw), ("min_frac", min_frac), ("max_frac", max_frac))
    overridden = [k for k, v in given if v is not None]
    if base is not None:
        label = base.label + (f" (modified: {', '.join(overridden)})" if overridden else "")
    else:
        label = ""
    try:
        plant = shift.Plant(
            base.name if base is not None and not overridden else "custom",
            float(avg_mw if avg_mw is not None else base.avg_mw),
            float(min_frac if min_frac is not None else (base.min_frac if base else 0.8)),
            float(max_frac if max_frac is not None else (base.max_frac if base else 1.2)),
            label,
        )
    except ValueError as exc:
        raise ToolError(str(exc)) from exc
    qh = pl.scan_parquet(_silver("energy_15min"))
    daily = pl.scan_parquet(_silver("energy_daily"))
    if start or end:
        a = _date(start, "start") if start else date(1900, 1, 1)
        b = _date(end, "end") if end else date.max
        qh = qh.filter(pl.col("date_berlin").is_between(a, b))
        daily = daily.filter(pl.col("date_berlin").is_between(a, b))
    days = shift.run(qh.collect(), daily.collect(), [plant])
    if days.height == 0:
        raise ToolError("no days with complete prices in that period")
    s = shift.summary(days).row(0, named=True)
    out = {
        "plant": plant.label or "custom plant",
        **({"plant_de": PRESET_DE[preset]} if preset in PRESET_DE else {}),
        "avg_mw": plant.avg_mw,
        "min_load_mw": _r(plant.avg_mw * plant.min_frac),
        "max_load_mw": _r(plant.avg_mw * plant.max_frac),
        "flexibility": f"{plant.min_frac:.0%}-{plant.max_frac:.0%} of average load",
        "period": f"{days['date_berlin'].min()} .. {days['date_berlin'].max()}",
        "days": s["days"],
        "energy_mwh": _r(float(s["energy_mwh"])),
        "savings_eur": _r(float(s["savings_eur"])),
        "savings_eur_per_mwh": _r(float(s["savings_eur_per_mwh"])),
        "savings_share_of_energy_cost": f"{s['savings_share']:.1%}",
        "savings_eur_per_year": _r(float(s["savings_eur_per_year"])),
        "caveat": "upper bound on the day-ahead energy cost only: no ramp limits, grid fees "
        "or intraday trading",
        "source": "gold/shift model on silver/energy_15min",
    }
    if start or end:
        last = days["date_berlin"].max()
        out |= _coverage(a if start else days["date_berlin"].min(), b if end else last)
    if s["days"] < 28:
        out["warning"] = (
            f"only {s['days']} days: savings_eur_per_year is extrapolated from a short period "
            "and not representative; omit start/end for the full history"
        )
    return out


def forecast_quality() -> dict:
    """Walk-forward backtest of the day-ahead price models (errors and price-range coverage)."""
    scores = pl.read_parquet(_gold("backtest_scores", "make backtest")).sort("mae")
    preds = pl.scan_parquet(_gold("backtest_predictions", "make backtest"))
    p = (
        preds.select(
            lo=pl.col("date_berlin").min(),
            hi=pl.col("date_berlin").max(),
            cov=((pl.col(PRICE) >= pl.col("p10")) & (pl.col(PRICE) <= pl.col("p90"))).mean(),
            cov_raw=(
                (pl.col(PRICE) >= pl.col("p10_raw")) & (pl.col(PRICE) <= pl.col("p90_raw"))
            ).mean(),
        )
        .collect()
        .row(0, named=True)
        if "p10_raw" in preds.collect_schema().names()
        else None
    )
    out = {
        "models_by_mae": [_row(r) for r in scores.iter_rows(named=True)],
        "metric_notes": "MAE/RMSE in EUR/MWh; skill_vs_naive_d1 = 1 - MAE / MAE of "
        "'same quarter-hour yesterday'",
        "source": "gold/backtest_scores, backtest_predictions",
    }
    imp_path = config.data_dir() / "gold" / "feature_importance.parquet"
    if imp_path.exists():
        imp = pl.read_parquet(imp_path).head(8)
        out["top_features_lightgbm"] = [
            {"feature": r["feature"], "share_of_gain_pct": _r(r["share"] * 100)}
            for r in imp.iter_rows(named=True)
        ]
        out["feature_notes"] = (
            "price_lag_Xd = price X days earlier at the same time; price_d1_* = yesterday's "
            "price stats; *_fc_* = TSO wind/PV forecasts for the delivery day; *_lag_2d/7d = "
            "load or generation 2/7 days earlier; doy/hour/weekday = calendar. These are the "
            "only inputs; there are no weather, fuel-price or storage features."
        )
    if p:
        out |= {
            "test_period": f"{p['lo']} .. {p['hi']}",
            "p10_p90_coverage_calibrated": f"{p['cov']:.0%}",
            "p10_p90_coverage_raw_quantiles": f"{p['cov_raw']:.0%}",
            "coverage_target": "80%",
        }
    return out


# --------------------------------------------------------------------------- registry


@dataclass(frozen=True)
class Tool:
    fn: Callable[..., dict]
    description: str
    parameters: dict

    @property
    def name(self) -> str:
        return self.fn.__name__

    def schema(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


DATE = {"type": "string", "description": "ISO date, e.g. 2026-03-01"}


def registry(search_docs: Callable[..., dict] | None = None) -> dict[str, Tool]:
    tools = [
        Tool(
            dataset_info,
            "Date range and freshness of the data. Call when unsure what exists.",
            {"type": "object", "properties": {}},
        ),
        Tool(
            price_stats,
            "Day-ahead price statistics for any date range: average base/peak/off-peak, min and "
            "max with time, total negative-price hours, daily spread. Give flat_mw to get the "
            "energy cost of a constant load (do not compute costs yourself).",
            {
                "type": "object",
                "properties": {
                    "start": DATE,
                    "end": DATE,
                    "flat_mw": {"type": "number", "description": "Constant load in MW (optional)."},
                },
                "required": ["start", "end"],
            },
        ),
        Tool(
            period_breakdown,
            "Prices per week (Mon-Sun) or per day within ONE range, with the cheapest and most "
            "expensive week/day identified. Use for 'which week/day was most expensive' and "
            "with flat_mw for the cost difference. To compare two periods (e.g. first vs "
            "second half of a month) call price_stats once for each period instead.",
            {
                "type": "object",
                "properties": {
                    "start": DATE,
                    "end": DATE,
                    "by": {"type": "string", "enum": ["week", "day"]},
                    "flat_mw": {"type": "number", "description": "Constant load in MW (optional)."},
                },
                "required": ["start", "end"],
            },
        ),
        Tool(
            monthly_market,
            "Monthly market facts: prices, peak minus off-peak, negative-price hours, renewable "
            "shares Germany vs Amprion (NRW), load-weighted price. For specific months pass "
            "months (e.g. ['2026-09']); there is no start/end. Use sort_by only for 'which "
            "month had the most/least ...' and report rank 1: most expensive/cheapest month = "
            "price_base, most negative hours = negative_price_hours, biggest peak/off-peak gap "
            "= peak_minus_offpeak, greenest = de_renewables_share or amp_renewables_share. "
            "Every ranked row has all columns: for 'the most expensive month and its gap', "
            "rank by price_base and read peak_minus_offpeak from rank 1.",
            {
                "type": "object",
                "properties": {
                    "months": {
                        "type": "array",
                        "items": {"type": "string", "description": "YYYY-MM"},
                        "description": "Months to return; omit for all months.",
                    },
                    "year": {"type": "integer", "description": "Only months of this year."},
                    "sort_by": {"type": "string", "enum": list(RANKABLE)},
                    "order": {
                        "type": "string",
                        "enum": ["highest", "lowest"],
                        "description": "highest (default) for most/maximum, lowest for least.",
                    },
                    "top": {
                        "type": "integer",
                        "description": "How many ranked months (default 3).",
                    },
                },
            },
        ),
        Tool(
            cheapest_hours,
            "Cheapest or most expensive hours of one day (incl. tomorrow once SMARD shows the "
            "auction results, usually by the evening), individually or as one block, e.g. to "
            "schedule a batch process.",
            {
                "type": "object",
                "properties": {
                    "day": DATE,
                    "hours": {"type": "integer", "description": "How many hours (1-23)."},
                    "contiguous": {"type": "boolean", "description": "One block in a row."},
                    "most_expensive": {"type": "boolean", "description": "Most expensive instead."},
                },
                "required": ["day"],
            },
        ),
        Tool(
            load_shift_savings,
            "Savings a flexible industrial plant gets by shifting load within each day to "
            "cheaper quarter-hours. The user's own plant: avg_mw plus min_frac/max_frac as share "
            "of average ('+/-20 %' = 0.8 / 1.2; '70 % to 120 %' = 0.7 / 1.2), no preset. Example "
            "plants: preset (generic_10pct, chlor_alkali, aluminium, steel_eaf); avg_mw/min_frac/"
            "max_frac given with a preset override it. A period the user mentions ('last 3 "
            "months') must be passed as start/end; omit start/end only if none is mentioned.",
            {
                "type": "object",
                "properties": {
                    "avg_mw": {"type": "number", "description": "Average load in MW."},
                    "min_frac": {
                        "type": "number",
                        "description": "Minimum load / average, e.g. 0.8.",
                    },
                    "max_frac": {
                        "type": "number",
                        "description": "Maximum load / average, e.g. 1.2.",
                    },
                    "preset": {
                        "type": "string",
                        "enum": ["generic_10pct", "chlor_alkali", "aluminium", "steel_eaf"],
                    },
                    "start": DATE,
                    "end": DATE,
                },
            },
        ),
        Tool(
            forecast_quality,
            "How well the price forecasting models perform (walk-forward backtest): MAE per "
            "model, skill vs naive, P10-P90 range coverage.",
            {"type": "object", "properties": {}},
        ),
    ]
    if search_docs is not None:
        tools.append(
            Tool(
                search_docs,
                "Search the project documentation for methods and definitions (how data is "
                "cleaned, what a column or model means, caveats). Not for numbers.",
                {
                    "type": "object",
                    "properties": {"query": {"type": "string"}, "k": {"type": "integer"}},
                    "required": ["query"],
                },
            )
        )
    return {t.name: t for t in tools}


def call(tools: dict[str, Tool], name: str, arguments: dict) -> dict:
    """Run a tool; never raises (errors go back to the model)."""
    tool = tools.get(name)
    if tool is None:
        return {"error": f"unknown tool {name!r}; available: {sorted(tools)}"}
    accepted = set(inspect.signature(tool.fn).parameters)
    given = {k: v for k, v in (arguments or {}).items() if v is not None}
    unknown = sorted(set(given) - accepted)
    if unknown:
        # Never drop arguments silently: a dropped filter (e.g. start/end) changes the answer.
        return {
            "error": f"{name} does not accept {unknown}; accepted arguments: {sorted(accepted)}. "
            "Call it again with accepted arguments only."
        }
    args = given
    try:
        return tool.fn(**args)
    except ToolError as exc:
        return {"error": str(exc)}
    except (TypeError, ValueError) as exc:
        return {"error": f"bad arguments for {name}: {exc}"}
