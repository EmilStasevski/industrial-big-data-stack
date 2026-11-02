"""Spark pipeline: identical results to the Polars pipeline (needs Java 17/21)."""

from __future__ import annotations

import shutil
from datetime import timedelta
from pathlib import Path

import polars as pl
import pytest

from ruhrwatt.silver import energy
from synthetic import END, PAD, START, _value, utc, write_bronze

pytestmark = [
    pytest.mark.spark,
    pytest.mark.skipif(shutil.which("java") is None, reason="Spark needs Java"),
]


@pytest.fixture(scope="module")
def spark():
    from ruhrwatt.spark.session import get_spark

    return get_spark("ruhrwatt-tests")


def _parity(spark, root: Path) -> dict:
    from ruhrwatt.spark import pipeline

    assert spark is not None  # module session is reused by getOrCreate
    return pipeline.parity(root)


def test_identical_on_the_autumn_dst_week(spark, tmp_path: Path) -> None:
    root = write_bronze(tmp_path / "b", START, END, padded_until=PAD)
    assert _parity(spark, root) == {
        "energy_15min": [],
        "energy_daily": [],
        "price_features_15min": [],
    }


def test_identical_on_the_spring_dst_week(spark, tmp_path: Path) -> None:
    root = write_bronze(tmp_path / "b", utc(2025, 3, 24), utc(2025, 4, 2))
    assert all(not d for d in _parity(spark, root).values())


def test_identical_with_holes_and_staggered_publication(spark, tmp_path: Path) -> None:
    """Nulls inside the price series (rolling min-samples logic), a missing forecast day,
    and prices published a day further than everything else."""
    hole = (utc(2025, 10, 22, 3), utc(2025, 10, 23, 9))  # 30 h without prices
    day = (utc(2025, 10, 24), utc(2025, 10, 25))

    def price(ts):
        return None if hole[0] <= ts < hole[1] else _value(energy.PRICE, ts)

    def fc_pv(ts):
        return None if day[0] <= ts < day[1] else _value("de_fc_pv_mwh", ts)

    root = write_bronze(
        tmp_path / "b", START, END, padded_until=PAD, override={"de_fc_pv_mwh": fc_pv}
    )
    p = write_bronze(
        tmp_path / "p",
        START,
        END + timedelta(days=1),
        padded_until=PAD,
        override={energy.PRICE: price},
    )
    from ruhrwatt.bronze.smard import bronze_path

    src = bronze_path("day_ahead_price", "DE-LU", "quarterhour", p)
    bronze_path("day_ahead_price", "DE-LU", "quarterhour", root).write_bytes(src.read_bytes())
    results = _parity(spark, root)
    assert results == {"energy_15min": [], "energy_daily": [], "price_features_15min": []}


def test_parity_detects_a_difference() -> None:
    from ruhrwatt.spark.parity import compare

    a = pl.DataFrame({"ts_utc": [1, 2, 3], "x": [1.0, 2.0, None], "f": [True, False, True]})
    b = pl.DataFrame({"ts_utc": [3, 2, 1], "x": [None, 2.0, 1.5], "f": [True, True, True]})
    diffs = {d.column: d.problem for d in compare(a, b)}
    assert set(diffs) == {"x", "f"}
    assert compare(a, a) == []
    assert compare(a, a.drop("f"))[0].column == "<columns>"


def test_build_writes_year_partitions_readable_by_polars(spark, tmp_path: Path) -> None:
    from ruhrwatt.spark import pipeline

    root = write_bronze(tmp_path / "b", utc(2025, 12, 29), utc(2026, 1, 3))
    out = pipeline.build(root, tmp_path / "spark")
    years = sorted(p.name for p in out["paths"]["15min"].glob("year=*"))
    assert years == ["year=2025", "year=2026"]
    back = pl.read_parquet(out["paths"]["15min"] / "**/*.parquet", hive_partitioning=True)
    assert (
        back.height
        == out["rows"]
        == pl.read_parquet(out["paths"]["features"] / "**/*.parquet", hive_partitioning=True).height
    )
    assert back["ts_utc"].is_unique().all()
    assert (back["year"] == back["date_berlin"].dt.year()).all()


def test_missing_bronze_gives_clear_error(spark, tmp_path: Path) -> None:
    from ruhrwatt.spark import silver

    root = write_bronze(tmp_path / "b", START, END, skip=("de_fc_pv_mwh",))
    with pytest.raises(FileNotFoundError, match="fc_pv/DE"):
        silver.build_15min(spark, root)


def test_cli_spark_commands(spark, tmp_path: Path, monkeypatch) -> None:
    from typer.testing import CliRunner

    from ruhrwatt.cli import app

    write_bronze(tmp_path / "bronze" / "smard", START, END, padded_until=PAD)
    monkeypatch.setenv("RUHRWATT_DATA_DIR", str(tmp_path))
    r = CliRunner().invoke(app, ["spark", "parity"])
    assert r.exit_code == 0, r.output
    assert r.output.count("[ok  ]") == 3
    r = CliRunner().invoke(app, ["spark", "build"])
    assert r.exit_code == 0, r.output
    assert "[FAIL]" not in r.output and (tmp_path / "spark" / "energy_daily").exists()


def test_identical_when_python_runs_in_berlin_time(spark, tmp_path: Path, monkeypatch) -> None:
    """Collecting Spark timestamps converts them to the local zone; nothing may depend on it."""
    import time

    monkeypatch.setenv("TZ", "Europe/Berlin")
    time.tzset()
    try:
        root = write_bronze(tmp_path / "b", START, END, padded_until=PAD)
        assert all(not d for d in _parity(spark, root).values())
    finally:
        monkeypatch.delenv("TZ")
        time.tzset()
