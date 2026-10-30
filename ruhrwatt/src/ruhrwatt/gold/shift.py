"""Load-shifting economics for a flexible industrial consumer.

Model (per Berlin day, day-ahead prices only):

* The plant runs at an average ``avg_mw``, so it needs ``avg_mw * hours`` MWh that day.
* Each quarter-hour it may draw between ``min_frac * avg_mw`` and ``max_frac * avg_mw``.
* It shifts consumption freely within the day (e.g. buffering intermediate products),
  keeping the daily energy fixed.

Minimise sum(price_t * energy_t) under those limits. Because every quarter-hour has the same
bounds, the exact optimum is greedy: everything at the minimum, then the remaining energy
into the cheapest quarter-hours at the maximum (one quarter-hour may be partial).

Not modelled: ramp limits, minimum run times, grid fees and levies, intraday trading.
Savings are therefore an upper bound on the wholesale-energy component only.
"""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl

from ruhrwatt.silver.energy import PRICE

QH_HOURS = 0.25


@dataclass(frozen=True)
class Plant:
    name: str
    avg_mw: float
    min_frac: float
    max_frac: float
    label: str = ""

    def __post_init__(self) -> None:
        if self.avg_mw <= 0:
            raise ValueError("avg_mw must be positive")
        if not 0 <= self.min_frac <= 1 <= self.max_frac:
            raise ValueError("need 0 <= min_frac <= 1 <= max_frac")


# Illustrative assumptions for typical NRW process types, NOT data of real plants.
# Change them (or use `ruhrwatt gold shift --avg-mw ...`) for a real case.
PRESETS: dict[str, Plant] = {
    p.name: p
    for p in (
        Plant("generic_10pct", 100, 0.90, 1.10, "Any plant, +/-10 % flexibility (reference)"),
        Plant("chlor_alkali", 60, 0.60, 1.15, "Chemicals: chlor-alkali electrolysis"),
        Plant("aluminium", 250, 0.85, 1.05, "Aluminium: smelter potlines"),
        Plant("steel_eaf", 120, 0.30, 1.60, "Steel: electric arc furnace, batch"),
    )
}


def priced_days(qh: pl.DataFrame, daily: pl.DataFrame) -> pl.DataFrame:
    """Quarter-hours of days whose day-ahead prices are complete (92/96/100 rows)."""
    ok = (
        qh.group_by("date_berlin")
        .agg(pl.len().alias("n"), pl.col(PRICE).null_count().alias("missing"))
        .join(daily.select("date_berlin", "expected_qh"), on="date_berlin")
        .filter((pl.col("n") == pl.col("expected_qh")) & (pl.col("missing") == 0))
        .select("date_berlin")
    )
    return qh.select("ts_utc", "ts_berlin", "date_berlin", PRICE).join(
        ok, on="date_berlin", how="semi"
    )


def schedule(prices: pl.DataFrame, plant: Plant) -> pl.DataFrame:
    """Optimal quarter-hour energy (MWh) next to the flat profile."""
    flat = plant.avg_mw * QH_HOURS
    floor = plant.min_frac * flat
    head = (plant.max_frac - plant.min_frac) * flat
    # Position of each quarter-hour in its day's price order (cheapest = 0, ties by time).
    rank = pl.int_range(pl.len()).over("date_berlin").cast(pl.Float64)
    flex_energy = (flat - floor) * pl.len().over("date_berlin")
    extra = (flex_energy - rank * head).clip(0, head)
    return (
        prices.sort("date_berlin", PRICE, "ts_utc")
        .with_columns(flat_mwh=pl.lit(flat), shifted_mwh=floor + extra)
        .sort("ts_utc")
    )


def daily_savings(sched: pl.DataFrame, plant: Plant) -> pl.DataFrame:
    p = pl.col(PRICE)
    return (
        sched.group_by("date_berlin")
        .agg(
            pl.len().alias("n_qh"),
            pl.col("flat_mwh").sum().alias("energy_mwh"),
            (p * pl.col("flat_mwh")).sum().alias("cost_flat_eur"),
            (p * pl.col("shifted_mwh")).sum().alias("cost_shifted_eur"),
            ((pl.col("shifted_mwh") - pl.col("flat_mwh")).abs().sum() / 2).alias("moved_mwh"),
            pl.col("shifted_mwh").filter(p < 0).sum().alias("negative_price_mwh"),
        )
        .with_columns(
            pl.lit(plant.name).alias("plant"),
            (pl.col("cost_flat_eur") - pl.col("cost_shifted_eur")).alias("savings_eur"),
        )
        .with_columns(
            (pl.col("savings_eur") / pl.col("energy_mwh")).alias("savings_eur_per_mwh"),
            (pl.col("cost_flat_eur") / pl.col("energy_mwh")).alias("price_flat_eur_mwh"),
            (pl.col("cost_shifted_eur") / pl.col("energy_mwh")).alias("price_shifted_eur_mwh"),
        )
        .sort("date_berlin")
        .select(
            "plant",
            "date_berlin",
            "n_qh",
            "energy_mwh",
            "cost_flat_eur",
            "cost_shifted_eur",
            "savings_eur",
            "savings_eur_per_mwh",
            "price_flat_eur_mwh",
            "price_shifted_eur_mwh",
            "moved_mwh",
            "negative_price_mwh",
        )
    )


def run(qh: pl.DataFrame, daily: pl.DataFrame, plants: list[Plant]) -> pl.DataFrame:
    """Daily savings for every plant, stacked."""
    prices = priced_days(qh, daily)
    return pl.concat([daily_savings(schedule(prices, p), p) for p in plants])


def summary(days: pl.DataFrame) -> pl.DataFrame:
    """Totals per plant over the whole period."""
    return (
        days.group_by("plant")
        .agg(
            pl.len().alias("days"),
            pl.col("energy_mwh").sum(),
            pl.col("cost_flat_eur").sum(),
            pl.col("savings_eur").sum(),
        )
        .with_columns(
            (pl.col("savings_eur") / pl.col("energy_mwh")).alias("savings_eur_per_mwh"),
            (pl.col("savings_eur") / pl.col("cost_flat_eur")).alias("savings_share"),
            (pl.col("savings_eur") / pl.col("days") * 365).alias("savings_eur_per_year"),
        )
        .sort("plant")
    )
