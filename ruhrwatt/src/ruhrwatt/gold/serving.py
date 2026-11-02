"""Serving snapshots: small JSON files the Go API (``api/``) serves.

Batch and serving are decoupled: the pipeline writes the snapshots, the API only reads
them (no Parquet or data libraries in the service, hot reload when files change).

``<data>/serving/``:

* ``monthly.json``: the monthly market mart (shares in %)
* ``prices.json``: 15-minute day-ahead prices of the last ``days`` days, incl. tomorrow once
  published, grouped by Berlin day
* ``freshness.json``: data coverage and generation time; written **last**, so a reader that
  sees a new freshness file also sees the new data files
"""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime, timedelta
from pathlib import Path

import polars as pl

from ruhrwatt import config
from ruhrwatt.download.smard import ATTRIBUTION
from ruhrwatt.silver.energy import PRICE

PRICE_DAYS = 31


def serving_dir(root: Path | None = None) -> Path:
    return root or (config.data_dir() / "serving")


def _write_json(path: Path, obj) -> None:
    tmp = path.with_name(path.name + ".part")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    os.replace(tmp, path)


def _utc(ts: datetime) -> str:
    return ts.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def export(
    silver_root: Path | None = None,
    gold_root: Path | None = None,
    out_root: Path | None = None,
    days: int = PRICE_DAYS,
) -> dict:
    silver = silver_root or (config.data_dir() / "silver")
    gold = gold_root or (config.data_dir() / "gold")
    out = serving_dir(out_root)
    out.mkdir(parents=True, exist_ok=True)

    qh = pl.scan_parquet(silver / "energy_15min.parquet")
    priced = qh.filter(pl.col(PRICE).is_not_null())
    last_price_day = priced.select(pl.col("date_berlin").max()).collect().item()
    prices = (
        priced.filter(pl.col("date_berlin") > last_price_day - timedelta(days=days))
        .select("ts_utc", "ts_berlin", "date_berlin", PRICE)
        .sort("ts_utc")
        .collect()
    )
    by_day: dict[str, list[dict]] = {}
    for r in prices.iter_rows(named=True):
        by_day.setdefault(r["date_berlin"].isoformat(), []).append(
            {
                "utc": _utc(r["ts_utc"]),
                "local": f"{r['ts_berlin']:%H:%M}",
                "price": round(r[PRICE], 2),
            }
        )

    monthly = pl.read_parquet(gold / "market_monthly.parquet")
    rows = []
    for r in monthly.sort("month").iter_rows(named=True):
        row = {}
        for k, v in r.items():
            if k == "month":
                row[k] = v.isoformat()[:7]
            elif isinstance(v, float):
                row[k] = round(v * 100, 2) if k.endswith("_share") else round(v, 2)
            else:
                row[k] = v
        rows.append(row)

    info = (
        qh.select(
            first=pl.col("date_berlin").min(),
            last_complete=pl.col("ts_utc").filter(pl.col("is_complete")).max(),
            last_price=pl.col("ts_utc").filter(pl.col(PRICE).is_not_null()).max(),
        )
        .collect()
        .row(0, named=True)
    )
    freshness = {
        "generated_at": _utc(datetime.now(UTC)),
        "first_day": info["first"].isoformat(),
        "last_complete_utc": _utc(info["last_complete"]),
        "last_price_utc": _utc(info["last_price"]),
        "price_days": sorted(by_day),
        "attribution": ATTRIBUTION,
    }
    _write_json(out / "monthly.json", {"months": rows})
    _write_json(out / "prices.json", {"unit": "EUR/MWh", "days": by_day})
    _write_json(out / "freshness.json", freshness)  # last: signals a complete snapshot
    return {"dir": out, "months": len(rows), "price_days": len(by_day), "freshness": freshness}
