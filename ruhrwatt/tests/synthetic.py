"""Synthetic bronze data shared by the silver and gold tests (offline, deterministic)."""

from __future__ import annotations

import math
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import polars as pl

from ruhrwatt.bronze.smard import bronze_path
from ruhrwatt.silver import energy

BERLIN = ZoneInfo("Europe/Berlin")


def _local(ts: datetime) -> datetime:
    return ts.astimezone(BERLIN)


def _value(column: str, ts: datetime) -> float:
    """Plausible synthetic values: PV only in daylight, Amprion = 30 % of Germany."""
    h = _local(ts).hour + _local(ts).minute / 60
    day_curve = max(0.0, math.sin(math.pi * (h - 6) / 12)) if 6 <= h <= 18 else 0.0
    if column == energy.PRICE:
        return round(80 + 60 * math.sin(math.pi * (h - 3) / 12) - 150 * day_curve**4, 2)
    t = ts.timestamp() / 3600  # hours: weather moves on multi-day cycles
    wind_on = 3_000 + 1_500 * math.sin(2 * math.pi * t / 37)
    wind_off = 1_500 + 700 * math.sin(2 * math.pi * t / 53)
    pv = 5_000 * day_curve
    base = {
        "load_mwh": 12_000 + 2_000 * day_curve,
        "residual_load_mwh": 6_000 - 3_000 * day_curve,
        "lignite_mwh": 2_000.0,
        "hard_coal_mwh": 800.0,
        "gas_mwh": 1_500.0,
        "wind_onshore_mwh": wind_on,
        "wind_offshore_mwh": wind_off,
        "pv_mwh": pv,
        "biomass_mwh": 1_000.0,
        "hydro_mwh": 400.0,
        # forecasts: close to the actuals, with a slow error
        "fc_wind_onshore_mwh": wind_on * (1 + 0.08 * math.sin(2 * math.pi * t / 11)),
        "fc_wind_offshore_mwh": wind_off * (1 - 0.05 * math.sin(2 * math.pi * t / 13)),
        "fc_pv_mwh": pv * 0.97,
    }[column.split("_", 1)[1]]
    return base * (0.3 if column.startswith("amp_") else 1.0)


def write_bronze(
    root: Path,
    start: datetime,
    end: datetime,
    *,
    padded_until: datetime | None = None,
    override: dict[str, Callable] | None = None,
    skip: tuple[str, ...] = (),
) -> Path:
    """Bronze files for every silver series between start and end (UTC, exclusive end).

    Rows from ``end`` to ``padded_until`` are null, like SMARD's padded current week.
    """
    override = override or {}
    stop = padded_until or end
    n = int((stop - start) / timedelta(minutes=15))
    stamps = [start + timedelta(minutes=15 * i) for i in range(n)]
    for s in energy.series():
        if s.column in skip:
            continue
        fn = override.get(s.column, lambda ts, c=s.column: _value(c, ts))
        values = [fn(ts) if ts < end else None for ts in stamps]
        out = bronze_path(s.dataset, s.region, "quarterhour", root)
        out.parent.mkdir(parents=True, exist_ok=True)
        pl.DataFrame(
            {
                "ts_utc": pl.Series(stamps, dtype=pl.Datetime("ms", "UTC")),
                "value": pl.Series(values, dtype=pl.Float64),
                "is_native_15min": [True] * n,
            }
        ).write_parquet(out)
    return root


def utc(*args: int) -> datetime:
    return datetime(*args, tzinfo=BERLIN).astimezone(UTC)


# Week with the autumn DST change (Sunday 2025-10-26 has 25 hours), padded like SMARD.
START = utc(2025, 10, 20)  # Monday 00:00 Berlin
END = utc(2025, 10, 27, 12)
PAD = utc(2025, 10, 27) + timedelta(days=7)
