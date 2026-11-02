"""Bronze SMARD series -> silver tables, with Polars (lazy, Rust-backed).

Two outputs in ``<data>/silver/``:

* ``energy_15min.parquet``: one row per quarter-hour (UTC grid), all 25 series as columns,
  Berlin calendar, NRW holidays, peak flags and derived energy/price features.
* ``energy_daily.parquet``: one row per Berlin day: base/peak/off-peak prices, spreads,
  negative-price quarter-hours, energy sums and shares.

SMARD quarter-hour values are energy per quarter-hour (MWh); multiply by 4 for average MW.
Rolling features look backward only, so the table can feed forecasting without leakage.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import holidays
import polars as pl

from ruhrwatt import config
from ruhrwatt.bronze.smard import bronze_path
from ruhrwatt.download import smard as dl

STEP = "15m"
STEP_MS = dl.STEP_MS["quarterhour"]
QH_PER_DAY = 96
TZ = "Europe/Berlin"

REGION_PREFIX = {"DE": "de", "Amprion": "amp"}
PRICE = "price_eur_mwh"

# dataset name -> column suffix (prefixed with the region, e.g. amp_load_mwh)
SERIES_COLUMN = {
    "grid_load": "load_mwh",
    "residual_load": "residual_load_mwh",
    "gen_lignite": "lignite_mwh",
    "gen_hard_coal": "hard_coal_mwh",
    "gen_gas": "gas_mwh",
    "gen_wind_onshore": "wind_onshore_mwh",
    "gen_wind_offshore": "wind_offshore_mwh",
    "gen_pv": "pv_mwh",
    "gen_biomass": "biomass_mwh",
    "gen_hydro": "hydro_mwh",
    "fc_wind_onshore": "fc_wind_onshore_mwh",
    "fc_wind_offshore": "fc_wind_offshore_mwh",
    "fc_pv": "fc_pv_mwh",
}
FOSSIL = ("lignite_mwh", "hard_coal_mwh", "gas_mwh")
WIND_PV = ("wind_onshore_mwh", "wind_offshore_mwh", "pv_mwh")
RENEWABLES = (*WIND_PV, "biomass_mwh", "hydro_mwh")
# Day-ahead TSO forecasts and the actual series each one predicts.
FORECAST_OF = {
    "fc_wind_onshore_mwh": "wind_onshore_mwh",
    "fc_wind_offshore_mwh": "wind_offshore_mwh",
    "fc_pv_mwh": "pv_mwh",
}
# Quantities that must be >= 0 and where a control area can never exceed Germany.
PHYSICAL = ("load_mwh", *FOSSIL, *RENEWABLES, *FORECAST_OF)


@dataclass(frozen=True)
class Series:
    dataset: str
    region: str
    column: str


def series() -> list[Series]:
    """Every bronze series silver needs, with its target column name."""
    out = [Series("day_ahead_price", "DE-LU", PRICE)]
    for dataset, suffix in SERIES_COLUMN.items():
        for region in dl.DATASETS[dataset].regions:
            out.append(Series(dataset, region, f"{REGION_PREFIX[region]}_{suffix}"))
    return out


def value_columns() -> list[str]:
    return [s.column for s in series()]


def region_columns(prefix: str, suffixes: tuple[str, ...] | list[str]) -> list[str]:
    """Columns that exist for this region (Amprion has no offshore wind)."""
    have = set(value_columns())
    return [f"{prefix}_{x}" for x in suffixes if f"{prefix}_{x}" in have]


def silver_dir(root: Path | None = None) -> Path:
    return root or (config.data_dir() / "silver")


# --------------------------------------------------------------------------- join


def _scan(s: Series, bronze_root: Path | None) -> pl.LazyFrame:
    return pl.scan_parquet(bronze_path(s.dataset, s.region, "quarterhour", bronze_root)).select(
        "ts_utc", pl.col("value").alias(s.column)
    )


def off_grid_rows(bronze_root: Path | None = None) -> int:
    """Bronze rows not on the 15-minute UTC grid (they cannot be joined)."""
    total = 0
    for s in series():
        total += (
            _scan(s, bronze_root)
            .filter(pl.col("ts_utc").dt.epoch("ms") % STEP_MS != 0)
            .select(pl.len())
            .collect()
            .item()
        )
    return total


def join_bronze(bronze_root: Path | None = None) -> pl.LazyFrame:
    """All series left-joined onto a continuous 15-minute grid, padding rows trimmed."""
    missing = [
        f"{s.dataset}/{s.region}"
        for s in series()
        if not bronze_path(s.dataset, s.region, "quarterhour", bronze_root).exists()
    ]
    if missing:
        raise FileNotFoundError(
            f"missing bronze series: {', '.join(missing)}; run `make bronze` first"
        )
    frames = [_scan(s, bronze_root) for s in series()]

    bounds = pl.concat(
        [f.select(lo=pl.col("ts_utc").min(), hi=pl.col("ts_utc").max()) for f in frames]
    ).collect()
    grid = pl.datetime_range(
        bounds["lo"].min(), bounds["hi"].max(), STEP, time_unit="ms", time_zone="UTC", eager=True
    )
    joined = pl.LazyFrame({"ts_utc": grid})
    for f in frames:
        joined = joined.join(f, on="ts_utc", how="left")

    # SMARD pads the current week with nulls; keep only rows where something was published.
    any_value = pl.any_horizontal(pl.col(value_columns()).is_not_null())
    first_last = joined.filter(any_value).select(
        lo=pl.col("ts_utc").min(), hi=pl.col("ts_utc").max()
    )
    return (
        joined.join(first_last, how="cross")
        .filter(pl.col("ts_utc").is_between(pl.col("lo"), pl.col("hi")))
        .drop("lo", "hi")
    )


# --------------------------------------------------------------------------- features


def nrw_holidays(years: range) -> list[date]:
    """Public holidays in North Rhine-Westphalia (incl. Fronleichnam, Allerheiligen)."""
    return sorted(holidays.country_holidays("DE", subdiv="NW", years=list(years)))


def _calendar(lf: pl.LazyFrame, holiday_dates: list[date]) -> pl.LazyFrame:
    local = pl.col("ts_utc").dt.convert_time_zone(TZ)
    lf = lf.with_columns(ts_berlin=local).with_columns(
        date_berlin=pl.col("ts_berlin").dt.date(),
        hour_berlin=pl.col("ts_berlin").dt.hour().cast(pl.Int8),
        weekday=pl.col("ts_berlin").dt.weekday().cast(pl.Int8),  # 1 = Monday
    )
    weekend = pl.col("weekday") >= 6
    holiday = pl.col("date_berlin").is_in(holiday_dates)
    return lf.with_columns(
        is_weekend=weekend,
        is_holiday_nrw=holiday,
        is_workday=~weekend & ~holiday,
        # EEX German peak: Mon-Fri 08:00-20:00 local time (holidays are not excluded).
        is_peak=~weekend & pl.col("hour_berlin").is_between(8, 19),
    )


def _energy(lf: pl.LazyFrame) -> pl.LazyFrame:
    exprs: list[pl.Expr] = []
    for p in REGION_PREFIX.values():
        load = pl.col(f"{p}_load_mwh")

        def total(suffixes: tuple[str, ...], p: str = p) -> pl.Expr:
            return pl.sum_horizontal(region_columns(p, suffixes), ignore_nulls=False)

        wind_pv, renewables, fossil = total(WIND_PV), total(RENEWABLES), total(FOSSIL)
        exprs += [
            wind_pv.alias(f"{p}_wind_pv_mwh"),
            renewables.alias(f"{p}_renewables_mwh"),
            fossil.alias(f"{p}_fossil_mwh"),
            total(tuple(FORECAST_OF)).alias(f"{p}_fc_wind_pv_mwh"),
            (wind_pv / load).alias(f"{p}_wind_pv_share"),
            (renewables / load).alias(f"{p}_renewables_share"),
            (fossil / load).alias(f"{p}_fossil_share"),
        ]
    exprs.append((pl.col("amp_load_mwh") / pl.col("de_load_mwh")).alias("amp_load_share_of_de"))
    return lf.with_columns(exprs)


def _price(lf: pl.LazyFrame) -> pl.LazyFrame:
    price = pl.col(PRICE)
    native_from = dl.DATASETS["day_ahead_price"].native_from_ms
    return lf.with_columns(
        price_is_native_15min=pl.col("ts_utc").dt.epoch("ms") >= native_from,
        price_is_negative=price < 0,
        price_delta_15min=price - price.shift(1),
        # Trailing 24 h window (current row included, nothing after it).
        price_mean_24h=price.rolling_mean(QH_PER_DAY, min_samples=QH_PER_DAY // 2),
        price_std_24h=price.rolling_std(QH_PER_DAY, min_samples=QH_PER_DAY // 2),
    )


def build_15min(bronze_root: Path | None = None) -> pl.DataFrame:
    lf = join_bronze(bronze_root).sort("ts_utc")
    years = lf.select(
        lo=pl.col("ts_utc").min().dt.year(), hi=pl.col("ts_utc").max().dt.year()
    ).collect()
    holiday_dates = nrw_holidays(range(years["lo"].item() - 1, years["hi"].item() + 2))
    lf = _price(_energy(_calendar(lf, holiday_dates)))
    lf = lf.with_columns(is_complete=pl.all_horizontal(pl.col(value_columns()).is_not_null()))
    head = [
        "ts_utc",
        "ts_berlin",
        "date_berlin",
        "hour_berlin",
        "weekday",
        "is_weekend",
        "is_holiday_nrw",
        "is_workday",
        "is_peak",
        "is_complete",
        PRICE,
        "price_is_native_15min",
    ]
    df = lf.collect()
    return df.select(head + [c for c in df.columns if c not in head])


# --------------------------------------------------------------------------- daily


def _sum(e: pl.Expr) -> pl.Expr:
    """Sum that is null (not 0) when a day has no values at all: missing data is not zero."""
    return pl.when(e.count() > 0).then(e.sum())


def build_daily(qh: pl.DataFrame) -> pl.DataFrame:
    """Aggregate the 15-minute table per Berlin calendar day."""
    price = pl.col(PRICE)
    peak = pl.col("is_peak")
    agg: list[pl.Expr] = [
        pl.len().alias("n_qh"),
        pl.col("is_complete").sum().alias("n_complete"),
        pl.col("weekday").first(),
        pl.col("is_weekend").first(),
        pl.col("is_holiday_nrw").first(),
        price.mean().alias("price_base"),
        price.filter(peak).mean().alias("price_peak"),
        price.filter(~peak).mean().alias("price_offpeak"),
        price.min().alias("price_min"),
        price.max().alias("price_max"),
        (price.max() - price.min()).alias("price_spread"),
        _sum(pl.col("price_is_negative")).alias("price_negative_qh"),
    ]
    for p in REGION_PREFIX.values():
        load = pl.col(f"{p}_load_mwh")
        paired = load.filter(price.is_not_null() & load.is_not_null())
        agg += [
            _sum(load).alias(f"{p}_load_mwh"),
            _sum(pl.col(f"{p}_residual_load_mwh")).alias(f"{p}_residual_load_mwh"),
            _sum(pl.col(f"{p}_wind_pv_mwh")).alias(f"{p}_wind_pv_mwh"),
            _sum(pl.col(f"{p}_renewables_mwh")).alias(f"{p}_renewables_mwh"),
            _sum(pl.col(f"{p}_fossil_mwh")).alias(f"{p}_fossil_mwh"),
            _sum(pl.col(f"{p}_fc_wind_pv_mwh")).alias(f"{p}_fc_wind_pv_mwh"),
            # What a consumer with this region's load profile paid on average.
            pl.when(paired.count() > 0)
            .then((price * load).sum() / paired.sum())
            .alias(f"{p}_load_weighted_price"),
        ]
    daily = qh.group_by("date_berlin").agg(agg).sort("date_berlin")

    midnight = pl.col("date_berlin").cast(pl.Datetime("ms")).dt.replace_time_zone(TZ)
    next_midnight = (
        (pl.col("date_berlin") + pl.duration(days=1))
        .cast(pl.Datetime("ms"))
        .dt.replace_time_zone(TZ)
    )
    expected = ((next_midnight - midnight).dt.total_minutes() // 15).cast(pl.UInt32)
    shares = []
    for p in REGION_PREFIX.values():
        shares += [
            (pl.col(f"{p}_wind_pv_mwh") / pl.col(f"{p}_load_mwh")).alias(f"{p}_wind_pv_share"),
            (pl.col(f"{p}_renewables_mwh") / pl.col(f"{p}_load_mwh")).alias(
                f"{p}_renewables_share"
            ),
            (pl.col(f"{p}_fossil_mwh") / pl.col(f"{p}_load_mwh")).alias(f"{p}_fossil_share"),
        ]
    # 92 / 96 / 100 quarter-hours on spring-forward / normal / fall-back days.
    complete = (pl.col("n_qh") == pl.col("expected_qh")) & (pl.col("n_complete") == pl.col("n_qh"))
    return daily.with_columns(expected_qh=expected).with_columns(
        complete.alias("is_complete_day"), *shares
    )


# --------------------------------------------------------------------------- write


def _write(df: pl.DataFrame, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".part")
    df.write_parquet(tmp, compression="zstd", statistics=True)
    os.replace(tmp, out)


def build(bronze_root: Path | None = None, silver_root: Path | None = None) -> dict:
    """Build both silver tables. Returns the frames and their paths."""
    qh = build_15min(bronze_root)
    daily = build_daily(qh)
    out = silver_dir(silver_root)
    paths = {"15min": out / "energy_15min.parquet", "daily": out / "energy_daily.parquet"}
    _write(qh, paths["15min"])
    _write(daily, paths["daily"])
    return {"qh": qh, "daily": daily, "paths": paths}
