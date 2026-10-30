"""Day-ahead price forecasting features (one row per delivery quarter-hour).

Information set: the DE-LU day-ahead auction for delivery day D closes on D-1 at 12:00
Berlin time. Every feature below is known before then:

* calendar of D (known in advance),
* prices of D-1, D-2 and D-7 (D-1 prices were set by the auction on D-2),
* load and generation actuals of D-2 and D-7 (D-1 actuals are not complete yet at 12:00),
* TSO day-ahead *forecasts* of wind and PV for day D itself.

No actual value (price, load, generation) from day D enters a feature, so the table can be
used for training and backtesting without look-ahead. Silver's rolling 24 h columns include
the current row and are therefore deliberately *not* used here.

Caveat on the forecasts: TSOs must publish day-ahead wind/PV forecasts by 18:00 on D-1,
which can be after the auction. Market participants have their own weather-based forecasts
of similar quality before noon, so the TSO forecast stands in for that information. A live
system would use a forecast vintage from before 12:00.

Lags are taken by Berlin wall-clock time ("same quarter-hour, k days earlier"), which is
what matters for load patterns. On the 25-hour autumn day the repeated 02:00-02:59 hour is
looked up once (first occurrence); on the 23-hour spring day the missing hour gives nulls.
"""

from __future__ import annotations

import math

import polars as pl

from ruhrwatt.silver.energy import PRICE

TARGET = PRICE
PRICE_LAG_DAYS = (1, 2, 7)
FUNDAMENTAL_LAG_DAYS = (2, 7)
FUNDAMENTALS = (
    "de_load_mwh",
    "de_residual_load_mwh",
    "de_wind_pv_mwh",
    "de_fossil_mwh",
    "amp_load_mwh",
    "amp_residual_load_mwh",
    "amp_wind_pv_mwh",
)
# Day-D forecasts used as they are (same quarter-hour), from the silver table.
FORECASTS = (
    "de_fc_wind_onshore_mwh",
    "de_fc_wind_offshore_mwh",
    "de_fc_pv_mwh",
    "de_fc_wind_pv_mwh",
    "amp_fc_wind_onshore_mwh",
    "amp_fc_pv_mwh",
    "amp_fc_wind_pv_mwh",
)
FORECAST_DERIVED = (
    "de_fc_wind_pv_day_mean",  # how windy/sunny day D is overall
    "de_fc_wind_pv_day_max",
    "de_fc_residual_proxy",  # load a week ago minus forecast wind + PV
    "de_fc_wind_pv_share_proxy",
)
CALENDAR = (
    "hour_berlin",
    "qh_of_day",
    "weekday",
    "month",
    "doy_sin",
    "doy_cos",
    "is_workday",
    "is_holiday_nrw",
    "is_peak",
)
WALL = "_wall"  # naive local timestamp used as the lag key


def lag_columns() -> list[str]:
    cols = [f"price_lag_{k}d" for k in PRICE_LAG_DAYS]
    cols += ["price_d1_mean", "price_d1_min", "price_d1_max", "price_d1_spread"]
    cols += [
        f"{c.removesuffix('_mwh')}_lag_{k}d" for c in FUNDAMENTALS for k in FUNDAMENTAL_LAG_DAYS
    ]
    return cols


def forecast_columns() -> list[str]:
    return [*FORECASTS, *FORECAST_DERIVED]


def feature_columns(forecasts: bool = True) -> list[str]:
    """Model inputs, in a fixed order. ``forecasts=False`` gives the history-only set."""
    return [*CALENDAR, *lag_columns(), *(forecast_columns() if forecasts else [])]


def _wall_lag(
    base: pl.DataFrame, lookup: pl.DataFrame, src: str, out: str, days: int
) -> pl.DataFrame:
    """Value of ``src`` at the same Berlin wall-clock time ``days`` earlier."""
    right = lookup.select(
        (pl.col(WALL) + pl.duration(days=days)).alias(WALL), pl.col(src).alias(out)
    )
    return base.join(right, on=WALL, how="left")


def build_features(qh: pl.DataFrame, daily: pl.DataFrame) -> pl.DataFrame:
    """Feature table from the silver 15-minute and daily tables."""
    base = qh.select(
        "ts_utc",
        "ts_berlin",
        "date_berlin",
        pl.col("ts_berlin").dt.replace_time_zone(None).alias(WALL),
        TARGET,
        "price_is_native_15min",
        "hour_berlin",
        (pl.col("hour_berlin").cast(pl.Int16) * 4 + pl.col("ts_berlin").dt.minute() // 15)
        .cast(pl.Int16)
        .alias("qh_of_day"),
        "weekday",
        pl.col("ts_berlin").dt.month().cast(pl.Int8).alias("month"),
        (2 * math.pi * pl.col("ts_berlin").dt.ordinal_day() / 365.25).sin().alias("doy_sin"),
        (2 * math.pi * pl.col("ts_berlin").dt.ordinal_day() / 365.25).cos().alias("doy_cos"),
        "is_workday",
        "is_holiday_nrw",
        "is_peak",
        *FORECASTS,
    )
    # One value per wall-clock time: the repeated autumn hour keeps its first occurrence.
    lookup = qh.select(
        pl.col("ts_berlin").dt.replace_time_zone(None).alias(WALL), TARGET, *FUNDAMENTALS
    ).unique(subset=WALL, keep="first", maintain_order=True)

    out = base
    for k in PRICE_LAG_DAYS:
        out = _wall_lag(out, lookup, TARGET, f"price_lag_{k}d", k)
    for c in FUNDAMENTALS:
        for k in FUNDAMENTAL_LAG_DAYS:
            out = _wall_lag(out, lookup, c, f"{c.removesuffix('_mwh')}_lag_{k}d", k)

    d1 = daily.select(
        (pl.col("date_berlin") + pl.duration(days=1)).cast(pl.Date).alias("date_berlin"),
        pl.col("price_base").alias("price_d1_mean"),
        pl.col("price_min").alias("price_d1_min"),
        pl.col("price_max").alias("price_d1_max"),
        pl.col("price_spread").alias("price_d1_spread"),
    )
    out = out.join(d1, on="date_berlin", how="left")
    fc = pl.col("de_fc_wind_pv_mwh")
    out = out.with_columns(
        fc.mean().over("date_berlin").alias("de_fc_wind_pv_day_mean"),
        fc.max().over("date_berlin").alias("de_fc_wind_pv_day_max"),
        (pl.col("de_load_lag_7d") - fc).alias("de_fc_residual_proxy"),
        (fc / pl.col("de_load_lag_7d")).alias("de_fc_wind_pv_share_proxy"),
    )
    head = ["ts_utc", "ts_berlin", "date_berlin", TARGET, "price_is_native_15min"]
    return out.sort("ts_utc").select(*head, *feature_columns())
