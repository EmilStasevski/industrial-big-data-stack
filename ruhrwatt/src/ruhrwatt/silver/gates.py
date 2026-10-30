"""Quality gates for the silver 15-minute table.

Errors mean the data (or a dataset label) is wrong and downstream work should stop.
Warnings are worth reading but normal for SMARD (for example a few missing values).
"""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl

from ruhrwatt.silver.energy import (
    FORECAST_OF,
    PHYSICAL,
    PRICE,
    REGION_PREFIX,
    region_columns,
    value_columns,
)

# Wide plausibility band for day-ahead prices in EUR/MWh: catches unit or parsing errors,
# not market events. Widen it if a real auction ever clears outside.
PRICE_MIN = -500.0
PRICE_MAX = 5000.0
# Hours (UTC) that are dark everywhere in Germany all year (21:00-01:59 UTC).
NIGHT_HOURS_UTC = (21, 22, 23, 0, 1)
PV_NIGHT_MAX_RATIO = 0.01  # night PV must be < 1 % of the all-day mean
TOLERANCE = 1.001  # Amprion <= DE, allowing rounding
FORECAST_MIN_CORR = 0.8  # a day-ahead forecast must track its actual series
# Amprion > DE in more than max(10 rows, 0.1 %) of rows is an error, fewer is a warning.
MAX_INCONSISTENT_ROWS = 10
MAX_INCONSISTENT_SHARE = 0.001


@dataclass(frozen=True)
class Gate:
    name: str
    ok: bool
    detail: str
    severity: str = "error"  # "error" or "warn"


def _count(df: pl.DataFrame, expr: pl.Expr) -> int:
    return int(df.select(expr.sum()).item() or 0)


def check(qh: pl.DataFrame, off_grid: int = 0) -> list[Gate]:
    gates: list[Gate] = []

    gates.append(Gate("on_grid", off_grid == 0, f"{off_grid} bronze rows off the 15-min grid"))

    ts = qh["ts_utc"]
    steps_ok = ts.is_sorted() and ts.n_unique() == len(ts)
    steps_ok = steps_ok and bool((ts.diff().drop_nulls().dt.total_minutes() == 15).all())
    gates.append(Gate("continuous_grid", steps_ok, "strictly increasing 15-minute timestamps"))

    price = pl.col(PRICE)
    bad = _count(qh, ~price.is_between(PRICE_MIN, PRICE_MAX) & price.is_not_null())
    lo, hi = qh[PRICE].min(), qh[PRICE].max()
    gates.append(
        Gate(
            "price_range",
            bad == 0,
            f"{bad} prices outside [{PRICE_MIN:g}, {PRICE_MAX:g}] EUR/MWh (observed {lo} .. {hi})",
        )
    )

    negatives = {
        c: n
        for p in REGION_PREFIX.values()
        for c in region_columns(p, PHYSICAL)
        if (n := _count(qh, pl.col(c) < 0))
    }
    gates.append(
        Gate("non_negative", not negatives, f"negative load/generation rows: {negatives or 0}")
    )

    # A swapped region or wrong id breaks thousands of rows; SMARD itself has the odd
    # inconsistent quarter-hour. Only a systematic share is an error.
    above: dict[str, int] = {}
    examples: list[str] = []
    both = [x for x in PHYSICAL if f"amp_{x}" in qh.columns and f"de_{x}" in qh.columns]
    for x in both:
        amp, de = pl.col(f"amp_{x}"), pl.col(f"de_{x}")
        rows = qh.filter(amp > de * TOLERANCE + 1)
        if rows.height:
            above[x] = rows.height
            r = rows.row(0, named=True)
            examples.append(
                f"{x} @ {r['ts_berlin']:%Y-%m-%d %H:%M}: "
                f"Amprion {r[f'amp_{x}']:g} > DE {r[f'de_{x}']:g}"
            )
    worst = max(above.values(), default=0)
    allowed = max(MAX_INCONSISTENT_ROWS, MAX_INCONSISTENT_SHARE * qh.height)
    detail = f"rows where Amprion > Germany: {above or 0}"
    if examples:
        detail += f" (e.g. {'; '.join(examples)})"
    gates.append(
        Gate(
            "amprion_within_de",
            not above,
            detail,
            severity="error" if worst > allowed else "warn",
        )
    )

    # Label sanity: if this fails, a PV filter id points at something else.
    night = pl.col("ts_utc").dt.hour().is_in(NIGHT_HOURS_UTC)
    for p in REGION_PREFIX.values():
        for name in region_columns(p, ("pv_mwh", "fc_pv_mwh")):
            col = pl.col(name)
            r = qh.select(night_mean=col.filter(night).mean(), all_mean=col.mean())
            night_mean, all_mean = r["night_mean"].item(), r["all_mean"].item()
            gate = f"{name.removesuffix('_mwh')}_dark_at_night"
            if not all_mean:
                gates.append(Gate(gate, False, "no PV values"))
                continue
            ratio = (night_mean or 0.0) / all_mean
            gates.append(
                Gate(
                    gate,
                    ratio < PV_NIGHT_MAX_RATIO,
                    f"night/all-day ratio {ratio:.4f} (must be < {PV_NIGHT_MAX_RATIO})",
                )
            )

    # Forecast label sanity: a wrong id would not correlate with the series it claims to predict.
    for p in REGION_PREFIX.values():
        for fc, actual in FORECAST_OF.items():
            if f"{p}_{fc}" not in qh.columns:
                continue
            pair = qh.select(f"{p}_{fc}", f"{p}_{actual}").drop_nulls()
            corr = (
                pair.select(pl.corr(f"{p}_{fc}", f"{p}_{actual}")).item()
                if pair.height > 2
                else None
            )
            ok = corr is not None and corr >= FORECAST_MIN_CORR
            shown = "n/a" if corr is None else f"{corr:.3f}"
            gates.append(
                Gate(
                    f"{p}_{fc.removesuffix('_mwh')}_tracks_actual",
                    ok,
                    f"correlation with {p}_{actual} = {shown} (must be >= {FORECAST_MIN_CORR})",
                )
            )

    # Missing values before the last complete row. After it, nulls are just publication lag:
    # prices are known a day ahead, load and generation arrive later.
    last_complete = qh.filter(pl.col("is_complete"))["ts_utc"].max()
    body = qh.filter(pl.col("ts_utc") <= last_complete) if last_complete else qh.clear()
    nulls = {c: n for c in value_columns() if (n := body[c].null_count())}
    gates.append(
        Gate(
            "internal_nulls",
            not nulls,
            f"missing values before the last complete quarter-hour: {nulls or 0}",
            severity="warn",
        )
    )
    return gates


def failed(gates: list[Gate]) -> list[Gate]:
    return [g for g in gates if not g.ok and g.severity == "error"]
