# PySpark: the same pipeline, written to scale out

`ruhrwatt spark build` (`make spark`) runs the two heavy data-preparation stages in Spark:
the silver tables (`energy_15min`, `energy_daily`) and the gold forecasting features. Model
training stays in Python: the boosting libraries are single-machine anyway, and the feature
table is small once built.

Requires Java 17 or 21 (`java -version`). Output in `data/spark/`, Hive-partitioned by
`year` (`energy_15min/year=2025/...`), readable by Polars, DuckDB and Spark alike.

## Parity with Polars

`ruhrwatt spark parity` (`make spark-parity`) builds every table with both engines from the
same bronze files and compares them column by column: same rows, same nulls, floats within
1e-9. The tests run this on the autumn and spring DST weeks, on data with a 30-hour price
gap, a missing forecast day and staggered publication, and with Python set to Berlin time.

The parity check found one real bug in the Polars pipeline: daily sums over a day with no
data were 0 instead of null (Polars sums all-null groups to 0, Spark to null). Polars now
returns null too, because missing data is not zero load.

One representational difference remains by design: Spark has no time-zone-aware timestamp,
so `ts_berlin` is a TIMESTAMP_NTZ with Berlin wall-clock time. On the repeated autumn hour
two rows share a `ts_berlin`; `ts_utc` stays the unique key.

## Spark-specific design

| Stage | Polars | Spark |
|---|---|---|
| read bronze | 25 lazy scans + 25 left joins | one scan of the Hive tree, then one conditional aggregation by integer series id (about 2x faster than `pivot`) |
| grid | `datetime_range` | `spark.range` over epoch seconds (distributed) |
| price lag / rolling 24 h | `shift`, `rolling_*` | window functions on a two-column (ts, price) frame, joined back |
| wall-clock lags | 17 joins | 3 self-joins (one per lag distance, all columns at once) |
| repeated autumn hour | `unique(keep="first")` | `row_number()` over the wall-clock key |
| holidays | `is_in` list | `isin` literal list (a few dozen dates) |

Bounds are collected as epoch seconds and years, never as timestamps: PySpark converts
collected timestamps to the Python process's local time zone, which would shift the grid
on a machine set to Berlin time (a test guards this).

The price lag and rolling statistics need one ordered pass over the whole series, which is a
single task in Spark. That is fine for one price series; with many bidding zones, partition
the window by zone.

## Benchmark (`make bench-spark`)

Synthetic bronze for all 25 series, growing history, both engines build and write all
three tables. 2-core cloud sandbox, 7 GB RAM; your numbers will differ:

| History | Bronze rows | Polars | Spark | Spark / Polars |
|---|---|---|---|---|
| 2 years (about today's data) | 1.8 M | 1.0 s | 43 s (+8 s JVM start) | 44x |
| 10 years | 8.8 M | 4.3 s | 59 s | 14x |
| 25 years | 21.9 M | 16.4 s | 113 s | 7x |

**Reading it:** on one machine, Polars wins at every size here; this data is small big
data, and Polars is the right default for it. Spark pays a fixed cost per stage
(scheduling, shuffles, JVM serialization), so its overhead shrinks relative to the work as
data grows, from 44x to 7x. Spark earns its place when the data outgrows one machine's
memory, or runs on a cluster (more zones, smart-meter data, many years of 1-minute data).
The point of the Spark version is that the same pipeline is ready for that, with results
proven identical.
