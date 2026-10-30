"""Gold tables on disk: features, load-shifting results and a monthly market mart.

``<data>/gold/``:

* ``price_features_15min.parquet``: forecasting features (see ``features.py``)
* ``load_shift_daily.parquet``: daily savings for each preset plant
* ``market_monthly.parquet``: one row per month, the table the assistant answers from
"""

from __future__ import annotations

import os
from pathlib import Path

import polars as pl

from ruhrwatt import config
from ruhrwatt.gold import features, shift
from ruhrwatt.silver.energy import silver_dir


def gold_dir(root: Path | None = None) -> Path:
    return root or (config.data_dir() / "gold")


def read_silver(silver_root: Path | None = None) -> tuple[pl.DataFrame, pl.DataFrame]:
    base = silver_dir(silver_root)
    paths = [base / "energy_15min.parquet", base / "energy_daily.parquet"]
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError(f"missing silver tables: {', '.join(missing)}; run `make silver`")
    return pl.read_parquet(paths[0]), pl.read_parquet(paths[1])


def market_monthly(daily: pl.DataFrame, shift_days: pl.DataFrame) -> pl.DataFrame:
    """Monthly market facts over complete days, plus savings per preset plant."""
    month = pl.col("date_berlin").dt.truncate("1mo").alias("month")
    full = daily.filter(pl.col("is_complete_day"))
    facts = (
        full.group_by(month)
        .agg(
            pl.len().alias("days"),
            pl.col("price_base").mean(),
            pl.col("price_peak").mean(),
            pl.col("price_offpeak").mean(),
            pl.col("price_min").min(),
            pl.col("price_max").max(),
            pl.col("price_spread").mean().alias("price_daily_spread_avg"),
            (pl.col("price_negative_qh").sum() / 4).alias("negative_price_hours"),
            (pl.col("price_negative_qh") > 0).sum().alias("days_with_negative_prices"),
            (pl.col("de_wind_pv_mwh").sum() / pl.col("de_load_mwh").sum()).alias(
                "de_wind_pv_share"
            ),
            (pl.col("amp_wind_pv_mwh").sum() / pl.col("amp_load_mwh").sum()).alias(
                "amp_wind_pv_share"
            ),
            (pl.col("de_renewables_mwh").sum() / pl.col("de_load_mwh").sum()).alias(
                "de_renewables_share"
            ),
            (pl.col("amp_renewables_mwh").sum() / pl.col("amp_load_mwh").sum()).alias(
                "amp_renewables_share"
            ),
            (pl.col("de_fossil_mwh").sum() / pl.col("de_load_mwh").sum()).alias("de_fossil_share"),
            (pl.col("amp_fossil_mwh").sum() / pl.col("amp_load_mwh").sum()).alias(
                "amp_fossil_share"
            ),
            (pl.col("amp_load_mwh").sum() / pl.col("de_load_mwh").sum()).alias(
                "amp_load_share_of_de"
            ),
            # Load-weighted price over the month = sum(price*load)/sum(load), from daily values.
            (
                (pl.col("de_load_weighted_price") * pl.col("de_load_mwh")).sum()
                / pl.col("de_load_mwh").sum()
            ).alias("de_load_weighted_price"),
            (
                (pl.col("amp_load_weighted_price") * pl.col("amp_load_mwh")).sum()
                / pl.col("amp_load_mwh").sum()
            ).alias("amp_load_weighted_price"),
        )
        .sort("month")
    )
    savings = (
        shift_days.group_by(month, "plant")
        .agg((pl.col("savings_eur").sum() / pl.col("energy_mwh").sum()).alias("eur_per_mwh"))
        .pivot(on="plant", index="month", values="eur_per_mwh")
        .rename(lambda c: c if c == "month" else f"shift_savings_eur_mwh_{c}")
    )
    return facts.join(savings, on="month", how="left").sort("month")


def _write(df: pl.DataFrame, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".part")
    df.write_parquet(tmp, compression="zstd", statistics=True)
    os.replace(tmp, out)


def build(
    silver_root: Path | None = None,
    gold_root: Path | None = None,
    plants: list[shift.Plant] | None = None,
) -> dict:
    qh, daily = read_silver(silver_root)
    plants = plants or list(shift.PRESETS.values())
    feats = features.build_features(qh, daily)
    shift_days = shift.run(qh, daily, plants)
    monthly = market_monthly(daily, shift_days)
    out = gold_dir(gold_root)
    paths = {
        "features": out / "price_features_15min.parquet",
        "shift": out / "load_shift_daily.parquet",
        "monthly": out / "market_monthly.parquet",
    }
    _write(feats, paths["features"])
    _write(shift_days, paths["shift"])
    _write(monthly, paths["monthly"])
    return {"features": feats, "shift": shift_days, "monthly": monthly, "paths": paths}
