# Bronze layer: SMARD

Path (Hive-style partitions, so Spark, Polars and DuckDB expose them as columns):

```
data/bronze/smard/dataset=<name>/region=<region>/resolution=<res>/part-0.parquet
```

| Column | Type | Notes |
|---|---|---|
| `ts_utc` | timestamp[ms, UTC], not null | Start of the interval. Strictly increasing. |
| `value` | double, nullable | SMARD nulls stay null (never 0). Negative prices are valid. |
| `is_native_15min` | bool, not null | False only for `day_ahead_price` rows before 2025-10-01 00:00 Europe/Berlin, where SMARD repeats the hourly price four times. |

## Rules enforced by `ruhrwatt-core`

1. Weekly files are parsed in parallel, sorted, and deduplicated by timestamp.
2. Identical duplicates collapse; a null next to a value keeps the value; two different
   values for one timestamp raise an error (nothing is silently picked).
3. Quality counters are returned for every run: `rows`, `nulls`, `duplicates`, `gaps`
   (missing quarter-hours or hours between first and last timestamp), `misaligned`
   (timestamps off the 15-minute or hourly grid), `native_rows`.
4. Files are written to `part-0.parquet.part` and renamed, so a crash never leaves a
   half-written file.

## Next layer

Silver joins these series and runs the cross-series quality gates: see `silver_schema.md`.
