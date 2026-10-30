"""Pure-Python reference implementation of the bronze step.

Two jobs: (1) the oracle the Rust core is tested against, (2) the honest baseline for the
benchmark. It does the same work as ``_core.build_bronze``: parse, sort, dedupe, count
quality, write Parquet (via pyarrow). Do not use it in pipelines.
"""

from __future__ import annotations

import json
from pathlib import Path

HOUR_MS = 3_600_000


def parse_week_py(path: str | Path) -> tuple[list[int], list[float | None]]:
    with open(path, "rb") as fh:
        series = json.load(fh)["series"]
    return [int(t) for t, _ in series], [None if v is None else float(v) for _, v in series]


def merge_py(rows: list[tuple[int, float | None]]):
    rows.sort(key=lambda r: r[0])
    ts: list[int] = []
    vals: list[float | None] = []
    duplicates = 0
    for t, v in rows:
        if ts and ts[-1] == t:
            duplicates += 1
            last = vals[-1]
            if last is not None and v is not None and last != v:
                raise ValueError(f"conflicting values for timestamp {t}: {last} vs {v}")
            if last is None and v is not None:
                vals[-1] = v
        else:
            ts.append(t)
            vals.append(v)
    return ts, vals, duplicates


def quality_py(ts, vals, step_ms: int, native_from_ms: int | None) -> dict:
    gaps = 0
    misaligned = 0
    if step_ms > 0:
        for a, b in zip(ts, ts[1:], strict=False):
            if b - a > step_ms:
                gaps += (b - a) // step_ms - 1
        if step_ms <= HOUR_MS:
            misaligned = sum(1 for t in ts if t % step_ms != 0)
    return {
        "rows": len(ts),
        "nulls": sum(1 for v in vals if v is None),
        "gaps": gaps,
        "misaligned": misaligned,
        "native_rows": len(ts)
        if native_from_ms is None
        else sum(1 for t in ts if t >= native_from_ms),
        "first_ts_ms": ts[0] if ts else None,
        "last_ts_ms": ts[-1] if ts else None,
    }


def write_parquet_py(path: str | Path, ts, vals, native_from_ms: int | None) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    schema = pa.schema(
        [
            pa.field("ts_utc", pa.timestamp("ms", tz="UTC"), nullable=False),
            pa.field("value", pa.float64(), nullable=True),
            pa.field("is_native_15min", pa.bool_(), nullable=False),
        ]
    )
    table = pa.Table.from_arrays(
        [
            pa.array(ts, type=pa.int64()).cast(pa.timestamp("ms", tz="UTC")),
            pa.array(vals, type=pa.float64()),
            pa.array([native_from_ms is None or t >= native_from_ms for t in ts], type=pa.bool_()),
        ],
        schema=schema,
    )
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path, compression="snappy")


def build_bronze_py(files, out_path, step_ms: int, native_from_ms: int | None = None) -> dict:
    """Same contract as ``ruhrwatt._core.build_bronze``."""
    if not files:
        raise ValueError("no input files")
    rows: list[tuple[int, float | None]] = []
    for f in files:
        ts_w, vals_w = parse_week_py(f)
        rows.extend(zip(ts_w, vals_w, strict=True))
    ts, vals, duplicates = merge_py(rows)
    stats = quality_py(ts, vals, step_ms, native_from_ms)
    stats["files"] = len(files)
    stats["duplicates"] = duplicates
    write_parquet_py(out_path, ts, vals, native_from_ms)
    return stats
