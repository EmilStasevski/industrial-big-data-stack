"""Rust core vs pure-Python reference, on synthetic SMARD files (offline)."""

from __future__ import annotations

import json
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from ruhrwatt import _core
from ruhrwatt.bronze import smard as bronze
from ruhrwatt.bronze.reference import build_bronze_py, parse_week_py
from ruhrwatt.download import smard as dl

BERLIN = ZoneInfo("Europe/Berlin")
QH = 900_000


def week_timestamps(monday: date) -> list[int]:
    """Quarter-hour epoch-ms stamps from Monday 00:00 to next Monday 00:00 Europe/Berlin."""
    nxt = monday + timedelta(days=7)
    start = datetime(monday.year, monday.month, monday.day, tzinfo=BERLIN)
    end = datetime(nxt.year, nxt.month, nxt.day, tzinfo=BERLIN)
    return list(range(int(start.timestamp() * 1000), int(end.timestamp() * 1000), QH))


def default_value(i: int, ts: int) -> float | None:
    if i % 97 == 5:
        return None  # SMARD pads missing values with null
    if i % 211 == 7:
        return -round(0.01 * (i % 90 + 1), 2)  # negative prices exist
    return round(35 + 0.37 * (i % 150) + (ts % 7) * 0.01, 2)


def write_week(raw_root: Path, dataset: str, region: str, monday: date, value_fn=default_value):
    ds = dl.DATASETS[dataset]
    stamps = week_timestamps(monday)
    series = [[t, value_fn(i, t)] for i, t in enumerate(stamps)]
    folder = dl.target_dir(raw_root, ds.filter_id, region, "quarterhour")
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{ds.filter_id}_{region}_quarterhour_{stamps[0]}.json"
    path.write_text(json.dumps({"meta_data": {"version": 1}, "series": series}))
    return path


def convert(tmp_path: Path, dataset: str, region: str) -> dict:
    return bronze.to_bronze(
        dataset, region, raw_root=tmp_path / "raw", bronze_root=tmp_path / "bronze"
    )


def read(tmp_path: Path, dataset: str, region: str) -> pa.Table:
    return pq.read_table(bronze.bronze_path(dataset, region, root=tmp_path / "bronze"))


DST_WEEKS = [date(2025, 10, 13), date(2025, 10, 20), date(2025, 10, 27)]


def test_dst_fall_back_week_has_676_rows_and_no_gaps(tmp_path) -> None:
    raw = tmp_path / "raw"
    # The week of 2025-10-20 contains the 25-hour day (2025-10-26): 672 + 4 quarter-hours.
    for monday in DST_WEEKS:
        write_week(raw, "grid_load", "DE", monday)
    stats = convert(tmp_path, "grid_load", "DE")
    assert len(week_timestamps(date(2025, 10, 20))) == 676
    assert stats["rows"] == 672 + 676 + 672
    assert stats["gaps"] == 0 and stats["duplicates"] == 0 and stats["misaligned"] == 0
    table = read(tmp_path, "grid_load", "DE")
    assert table.schema.field("ts_utc").type == pa.timestamp("ms", tz="UTC")
    ts = table.column("ts_utc").cast(pa.int64()).to_pylist()
    assert all(b - a == QH for a, b in zip(ts, ts[1:], strict=False))


def test_spring_forward_week_has_fewer_rows(tmp_path) -> None:
    raw = tmp_path / "raw"
    write_week(raw, "grid_load", "DE", date(2025, 3, 24))  # contains the 23-hour day
    stats = convert(tmp_path, "grid_load", "DE")
    assert stats["rows"] == 672 - 4
    assert stats["gaps"] == 0


def test_nulls_and_negative_prices_survive(tmp_path) -> None:
    write_week(tmp_path / "raw", "day_ahead_price", "DE-LU", date(2025, 11, 3))
    stats = convert(tmp_path, "day_ahead_price", "DE-LU")
    col = read(tmp_path, "day_ahead_price", "DE-LU").column("value")
    assert stats["nulls"] == col.null_count > 0
    assert min(v for v in col.to_pylist() if v is not None) < 0


def test_rust_output_equals_python_reference(tmp_path) -> None:
    raw = tmp_path / "raw"
    for monday in DST_WEEKS:
        write_week(raw, "day_ahead_price", "DE-LU", monday)
    files = bronze.raw_files("day_ahead_price", "DE-LU", root=raw)
    native = dl.DATASETS["day_ahead_price"].native_from_ms
    out_rs, out_py = tmp_path / "rs.parquet", tmp_path / "py.parquet"

    stats_rs = _core.build_bronze([str(f) for f in files], str(out_rs), QH, native)
    stats_py = build_bronze_py(files, out_py, QH, native)

    assert stats_rs == stats_py
    assert pq.read_table(out_rs).equals(pq.read_table(out_py))


def test_parse_week_matches_reference(tmp_path) -> None:
    path = write_week(tmp_path / "raw", "grid_load", "DE", date(2025, 11, 10))
    assert _core.parse_week(str(path)) == parse_week_py(path)


def test_missing_week_is_counted_as_gap(tmp_path) -> None:
    raw = tmp_path / "raw"
    write_week(raw, "grid_load", "DE", date(2025, 11, 3))
    write_week(raw, "grid_load", "DE", date(2025, 11, 17))  # 11-10 missing
    assert convert(tmp_path, "grid_load", "DE")["gaps"] == 672


def test_native_flag_switches_at_the_15min_cutoff(tmp_path) -> None:
    raw = tmp_path / "raw"
    write_week(raw, "day_ahead_price", "DE-LU", date(2025, 9, 29))
    stats = convert(tmp_path, "day_ahead_price", "DE-LU")
    cutoff = dl.DAY_AHEAD_NATIVE_15MIN_FROM_MS
    table = read(tmp_path, "day_ahead_price", "DE-LU")
    ts = table.column("ts_utc").cast(pa.int64()).to_pylist()
    flags = table.column("is_native_15min").to_pylist()
    assert flags == [t >= cutoff for t in ts]
    assert stats["native_rows"] == sum(flags)
    assert 0 < sum(flags) < len(flags)


def test_other_datasets_are_always_native(tmp_path) -> None:
    write_week(tmp_path / "raw", "gen_pv", "DE", date(2025, 6, 2))
    convert(tmp_path, "gen_pv", "DE")
    table = read(tmp_path, "gen_pv", "DE")
    assert all(table.column("is_native_15min").to_pylist())


def test_identical_overlap_is_deduplicated(tmp_path) -> None:
    raw = tmp_path / "raw"
    path = write_week(raw, "grid_load", "DE", date(2025, 11, 3))
    copy = path.with_name(path.name.replace("quarterhour_", "quarterhour_9"))  # distinct name
    copy.write_bytes(path.read_bytes())
    stats = convert(tmp_path, "grid_load", "DE")
    assert stats["rows"] == 672 and stats["duplicates"] == 672


def test_conflicting_duplicate_values_raise(tmp_path) -> None:
    raw = tmp_path / "raw"
    path = write_week(raw, "grid_load", "DE", date(2025, 11, 3))
    payload = json.loads(path.read_text())
    payload["series"][10][1] = 12345.0
    other = path.with_name(path.name.replace("quarterhour_", "quarterhour_9"))
    other.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="conflicting values"):
        convert(tmp_path, "grid_load", "DE")


def test_malformed_file_raises_value_error(tmp_path) -> None:
    path = write_week(tmp_path / "raw", "grid_load", "DE", date(2025, 11, 3))
    path.write_text('{"nope": 1}')
    with pytest.raises(ValueError, match="invalid SMARD JSON"):
        convert(tmp_path, "grid_load", "DE")


def test_missing_raw_files_give_clear_error(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="smard download"):
        convert(tmp_path, "grid_load", "DE")
