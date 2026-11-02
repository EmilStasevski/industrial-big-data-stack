"""Compare a Spark result with the Polars result, column by column (pure Polars)."""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl

REL_TOL = 1e-9
ABS_TOL = 1e-9


@dataclass(frozen=True)
class Diff:
    column: str
    problem: str


def _normalize(df: pl.DataFrame) -> pl.DataFrame:
    """Same representation for both engines: naive Berlin wall time, us timestamps, f64 numbers."""
    out = []
    for name, dtype in df.schema.items():
        c = pl.col(name)
        if isinstance(dtype, pl.Datetime):
            if dtype.time_zone not in (None, "UTC"):
                c = c.dt.replace_time_zone(None)  # tz-aware Berlin -> wall-clock
            out.append(c.dt.cast_time_unit("us"))
        elif dtype.is_numeric():
            out.append(c.cast(pl.Float64))
        else:
            out.append(c)
    return df.select(out)


def compare(polars_df: pl.DataFrame, spark_df: pl.DataFrame, key: str = "ts_utc") -> list[Diff]:
    """Empty list = identical (floats within 1e-9)."""
    diffs: list[Diff] = []
    a, b = _normalize(polars_df), _normalize(spark_df)
    if set(a.columns) != set(b.columns):
        diffs.append(
            Diff(
                "<columns>",
                f"only polars: {sorted(set(a.columns) - set(b.columns))}, "
                f"only spark: {sorted(set(b.columns) - set(a.columns))}",
            )
        )
        common = [c for c in a.columns if c in b.columns]
        a, b = a.select(common), b.select(common)
    if a.height != b.height:
        return [*diffs, Diff("<rows>", f"polars {a.height} rows, spark {b.height} rows")]
    a, b = a.sort(key), b.sort(key)
    for c in a.columns:
        x, y = a[c], b[c]
        if not (x.is_null() == y.is_null()).all():
            n = int((x.is_null() != y.is_null()).sum())
            diffs.append(Diff(c, f"{n} rows null in one engine only"))
            continue
        both = ~x.is_null()
        x, y = x.filter(both), y.filter(both)
        if x.dtype == pl.Float64:
            tol = ABS_TOL + REL_TOL * y.abs()
            bad = int(((x - y).abs() > tol).sum())
            if bad:
                worst = float((x - y).abs().max())
                diffs.append(Diff(c, f"{bad} values differ (max abs diff {worst:.3g})"))
        elif not (x == y).all():
            diffs.append(Diff(c, f"{int((x != y).sum())} values differ"))
    return diffs
