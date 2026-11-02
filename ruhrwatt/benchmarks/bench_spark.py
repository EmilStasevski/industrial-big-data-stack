"""Polars vs Spark for the silver + gold-feature stages, on synthetic data of growing size.

Synthetic bronze for all 25 series is generated vectorized (numpy) for N years ending
2026-10-01, then both engines build the 15-min table, the daily table and the forecasting
features and write them to Parquet. Spark's JVM start-up is reported separately.

    uv run python benchmarks/bench_spark.py --years 2 10 25
"""

from __future__ import annotations

import argparse
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import polars as pl

from ruhrwatt.bronze.smard import bronze_path
from ruhrwatt.silver import energy

END = datetime(2026, 10, 1, tzinfo=UTC)
STEP = 900


def make_bronze(root: Path, years: int, seed: int = 0) -> int:
    rng = np.random.default_rng(seed)
    end_s = int(END.timestamp())
    start_s = end_s - years * 365 * 86_400
    t = np.arange(start_s, end_s, STEP, dtype=np.int64)
    hours = (t % 86_400) / 3600 + 1  # rough CET
    day = np.clip(np.sin(np.pi * (hours - 6) / 12), 0, None) * ((hours > 6) & (hours < 18))
    th = t / 3600
    wind = 3000 + 1500 * np.sin(2 * np.pi * th / 37)
    base = {
        "load_mwh": 12_000 + 2_000 * day,
        "residual_load_mwh": 6_000 - 3_000 * day,
        "lignite_mwh": np.full_like(th, 2000.0),
        "hard_coal_mwh": np.full_like(th, 800.0),
        "gas_mwh": np.full_like(th, 1500.0),
        "wind_onshore_mwh": wind,
        "wind_offshore_mwh": 1500 + 700 * np.sin(2 * np.pi * th / 53),
        "pv_mwh": 5000 * day,
        "biomass_mwh": np.full_like(th, 1000.0),
        "hydro_mwh": np.full_like(th, 400.0),
        "fc_wind_onshore_mwh": wind * 1.03,
        "fc_wind_offshore_mwh": (1500 + 700 * np.sin(2 * np.pi * th / 53)) * 0.97,
        "fc_pv_mwh": 4900 * day,
    }
    ts = pl.Series("ts_utc", t * 1000).cast(pl.Datetime("ms", "UTC"))
    for s in energy.series():
        if s.column == energy.PRICE:
            v = 220 - 0.025 * wind + rng.normal(0, 12, len(t))
        else:
            v = base[s.column.split("_", 1)[1]] * (0.3 if s.column.startswith("amp_") else 1)
        out = bronze_path(s.dataset, s.region, "quarterhour", root)
        out.parent.mkdir(parents=True, exist_ok=True)
        pl.DataFrame({"ts_utc": ts, "value": v, "is_native_15min": True}).write_parquet(out)
    return len(t) * len(energy.series())


def run_polars(root: Path, out: Path) -> float:
    from ruhrwatt.gold import features

    t0 = time.perf_counter()
    qh = energy.build_15min(root)
    daily = energy.build_daily(qh)
    feats = features.build_features(qh, daily)
    out.mkdir(parents=True, exist_ok=True)
    qh.write_parquet(out / "qh.parquet")
    daily.write_parquet(out / "daily.parquet")
    feats.write_parquet(out / "features.parquet")
    return time.perf_counter() - t0


def run_spark(root: Path, out: Path) -> tuple[float, float]:
    from ruhrwatt.spark import pipeline

    r = pipeline.build(root, out)
    return r["startup_s"], r["run_s"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", type=int, nargs="+", default=[2, 10, 25])
    args = ap.parse_args()
    rows = []
    for y in args.years:
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            n = make_bronze(tmp / "bronze", y)
            p = run_polars(tmp / "bronze", tmp / "polars")
            start, s = run_spark(tmp / "bronze", tmp / "spark")
            rows.append((y, n, p, s, start))
            print(
                f"{y:>3} years, {n / 1e6:5.1f} M bronze rows: polars {p:6.1f} s | "
                f"spark {s:6.1f} s (+{start:.0f} s JVM start) | spark/polars {s / p:4.1f}x",
                flush=True,
            )


if __name__ == "__main__":
    main()
