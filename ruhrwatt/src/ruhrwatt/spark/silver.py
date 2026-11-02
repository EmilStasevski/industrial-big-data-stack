"""Bronze -> silver in Spark, column for column the same as ``ruhrwatt.silver.energy``.

Spark-native choices (they matter once the data no longer fits one machine):

* one scan of the whole Hive-partitioned bronze tree, turned wide with one conditional
  aggregation (faster than ``pivot``), instead of 25 separate reads and joins;
* the continuous 15-minute grid comes from ``spark.range`` (distributed, no driver list);
* holidays are a tiny literal list, prices' lag and rolling statistics are window functions.

Difference to Polars: Spark has no time-zone-aware timestamp type, so ``ts_berlin`` is a
TIMESTAMP_NTZ holding Berlin wall-clock time. On the repeated autumn hour two rows share
the same ``ts_berlin``; ``ts_utc`` stays the unique key.
"""

from __future__ import annotations

from pathlib import Path

from pyspark.sql import Column, DataFrame, SparkSession, Window
from pyspark.sql import functions as F

from ruhrwatt.bronze.smard import bronze_path
from ruhrwatt.download import smard as dl
from ruhrwatt.silver import energy as E

TZ = E.TZ
STEP_S = E.STEP_MS // 1000


def bronze_root(root: Path | None = None) -> Path:
    from ruhrwatt import config

    return root or (config.data_dir() / "bronze" / "smard")


def _check_present(root: Path) -> None:
    missing = [
        f"{s.dataset}/{s.region}"
        for s in E.series()
        if not bronze_path(s.dataset, s.region, "quarterhour", root).exists()
    ]
    if missing:
        raise FileNotFoundError(f"missing bronze series: {', '.join(missing)}; run `make bronze`")


def read_wide(spark: SparkSession, root: Path) -> DataFrame:
    """All bronze series in one scan, turned wide: one column per series.

    Conditional aggregation over an integer series id instead of ``pivot``: about twice as
    fast in local benchmarks (each (ts, series) pair occurs once, so max == the value).
    """
    _check_present(root)
    series = E.series()
    mapping = spark.createDataFrame(
        [(s.dataset, s.region, i) for i, s in enumerate(series)],
        "dataset string, region string, idx int",
    )
    long = (
        spark.read.option("basePath", str(root))
        .parquet(str(root))
        .where(F.col("resolution") == "quarterhour")
        .join(F.broadcast(mapping), ["dataset", "region"])
        .select("ts_utc", "idx", "value")
    )
    return long.groupBy("ts_utc").agg(
        *[
            F.max(F.when(F.col("idx") == i, F.col("value"))).alias(s.column)
            for i, s in enumerate(series)
        ]
    )


def _grid(spark: SparkSession, wide: DataFrame) -> tuple[DataFrame, tuple[int, int]]:
    """Continuous 15-minute grid over the bronze range, left-joined, padding trimmed.

    Also returns the first and last year, so no second pass is needed for the holidays.
    """
    any_value = F.greatest(*[F.col(c).isNotNull() for c in E.value_columns()])
    # Fetch bounds as epoch seconds and UTC years: collecting timestamps would convert them
    # to the Python process's local time zone (e.g. Berlin) and shift the whole grid.
    stamp = F.when(any_value, F.col("ts_utc"))
    b = wide.agg(
        F.min(F.unix_seconds(stamp)).alias("lo"),
        F.max(F.unix_seconds(stamp)).alias("hi"),
        F.min(F.year(stamp)).alias("first_year"),
        F.max(F.year(stamp)).alias("last_year"),
    ).first()
    grid = spark.range(b["lo"], b["hi"] + 1, STEP_S).select(
        F.timestamp_seconds("id").alias("ts_utc")
    )
    return grid.join(wide, "ts_utc", "left"), (b["first_year"], b["last_year"])


def _calendar(df: DataFrame, holidays: list) -> DataFrame:
    local = F.from_utc_timestamp("ts_utc", TZ).cast("timestamp_ntz")
    df = df.withColumn("ts_berlin", local)
    df = df.withColumns(
        {
            "date_berlin": F.to_date("ts_berlin"),
            "hour_berlin": F.hour("ts_berlin").cast("tinyint"),
            # Spark: 1 = Sunday; Polars/ISO: 1 = Monday
            "weekday": (((F.dayofweek("ts_berlin") + 5) % 7) + 1).cast("tinyint"),
        }
    )
    weekend = F.col("weekday") >= 6
    holiday = F.col("date_berlin").isin(holidays)
    return df.withColumns(
        {
            "is_weekend": weekend,
            "is_holiday_nrw": holiday,
            "is_workday": ~weekend & ~holiday,
            "is_peak": ~weekend & F.col("hour_berlin").between(8, 19),
        }
    )


def _total(prefix: str, suffixes: tuple[str, ...]) -> Column:
    cols = E.region_columns(prefix, suffixes)
    out = F.col(cols[0])
    for c in cols[1:]:
        out = out + F.col(c)  # null if any part is null, like Polars ignore_nulls=False
    return out


def _energy(df: DataFrame) -> DataFrame:
    new: dict[str, Column] = {}
    for p in E.REGION_PREFIX.values():
        load = F.col(f"{p}_load_mwh")
        wind_pv, ren, fossil = (_total(p, E.WIND_PV), _total(p, E.RENEWABLES), _total(p, E.FOSSIL))
        new |= {
            f"{p}_wind_pv_mwh": wind_pv,
            f"{p}_renewables_mwh": ren,
            f"{p}_fossil_mwh": fossil,
            f"{p}_fc_wind_pv_mwh": _total(p, tuple(E.FORECAST_OF)),
            f"{p}_wind_pv_share": wind_pv / load,
            f"{p}_renewables_share": ren / load,
            f"{p}_fossil_share": fossil / load,
        }
    new["amp_load_share_of_de"] = F.col("amp_load_mwh") / F.col("de_load_mwh")
    return df.withColumns(new)


def _price(df: DataFrame) -> DataFrame:
    price = F.col(E.PRICE)
    # Lag and trailing 24 h statistics need one ordered pass over the whole series. Do it on
    # a two-column frame (ts, price) and join back, so only the price column goes through the
    # single ordered task, not all 55 columns.
    order = Window.orderBy("ts_utc")
    trailing = order.rowsBetween(-(E.QH_PER_DAY - 1), 0)
    enough = F.count(price).over(trailing) >= E.QH_PER_DAY // 2
    rolled = df.select("ts_utc", E.PRICE).select(
        "ts_utc",
        (price - F.lag(price).over(order)).alias("price_delta_15min"),
        F.when(enough, F.avg(price).over(trailing)).alias("price_mean_24h"),
        F.when(enough, F.stddev_samp(price).over(trailing)).alias("price_std_24h"),
    )
    native_from = dl.DATASETS["day_ahead_price"].native_from_ms
    return df.withColumns(
        {
            "price_is_native_15min": F.unix_millis("ts_utc") >= native_from,
            "price_is_negative": price < 0,
        }
    ).join(rolled, "ts_utc", "left")


def build_15min(spark: SparkSession, root: Path | None = None) -> DataFrame:
    root = bronze_root(root)
    df, (first_year, last_year) = _grid(spark, read_wide(spark, root))
    holidays = E.nrw_holidays(range(first_year - 1, last_year + 2))
    df = _price(_energy(_calendar(df, holidays)))
    df = df.withColumn("is_complete", F.least(*[F.col(c).isNotNull() for c in E.value_columns()]))
    head = [
        "ts_utc",
        "ts_berlin",
        "date_berlin",
        "hour_berlin",
        "weekday",
        "is_weekend",
        "is_holiday_nrw",
        "is_workday",
        "is_peak",
        "is_complete",
        E.PRICE,
        "price_is_native_15min",
    ]
    rest = [c for c in df.columns if c not in head]
    return df.select(*head, *rest)


def build_daily(qh: DataFrame) -> DataFrame:
    price = F.col(E.PRICE)
    peak = F.col("is_peak")
    agg: list[Column] = [
        F.count(F.lit(1)).alias("n_qh"),
        F.sum(F.col("is_complete").cast("int")).alias("n_complete"),
        F.min("weekday").alias("weekday"),
        F.min("is_weekend").alias("is_weekend"),
        F.min("is_holiday_nrw").alias("is_holiday_nrw"),
        F.avg(price).alias("price_base"),
        F.avg(F.when(peak, price)).alias("price_peak"),
        F.avg(F.when(~peak, price)).alias("price_offpeak"),
        F.min(price).alias("price_min"),
        F.max(price).alias("price_max"),
        (F.max(price) - F.min(price)).alias("price_spread"),
        F.sum(F.col("price_is_negative").cast("int")).alias("price_negative_qh"),
    ]
    for p in E.REGION_PREFIX.values():
        load = F.col(f"{p}_load_mwh")
        both = price.isNotNull() & load.isNotNull()
        agg += [
            F.sum(load).alias(f"{p}_load_mwh"),
            F.sum(f"{p}_residual_load_mwh").alias(f"{p}_residual_load_mwh"),
            F.sum(f"{p}_wind_pv_mwh").alias(f"{p}_wind_pv_mwh"),
            F.sum(f"{p}_renewables_mwh").alias(f"{p}_renewables_mwh"),
            F.sum(f"{p}_fossil_mwh").alias(f"{p}_fossil_mwh"),
            F.sum(f"{p}_fc_wind_pv_mwh").alias(f"{p}_fc_wind_pv_mwh"),
            (F.sum(price * load) / F.sum(F.when(both, load))).alias(f"{p}_load_weighted_price"),
        ]
    daily = qh.groupBy("date_berlin").agg(*agg)
    midnight = F.unix_seconds(F.to_utc_timestamp(F.col("date_berlin").cast("timestamp"), TZ))
    next_midnight = F.unix_seconds(
        F.to_utc_timestamp(F.date_add("date_berlin", 1).cast("timestamp"), TZ)
    )
    daily = daily.withColumn("expected_qh", ((next_midnight - midnight) / STEP_S).cast("int"))
    shares: dict[str, Column] = {
        "is_complete_day": (F.col("n_qh") == F.col("expected_qh"))
        & (F.col("n_complete") == F.col("n_qh"))
    }
    for p in E.REGION_PREFIX.values():
        load = F.col(f"{p}_load_mwh")
        shares |= {
            f"{p}_wind_pv_share": F.col(f"{p}_wind_pv_mwh") / load,
            f"{p}_renewables_share": F.col(f"{p}_renewables_mwh") / load,
            f"{p}_fossil_share": F.col(f"{p}_fossil_mwh") / load,
        }
    return daily.withColumns(shares).orderBy("date_berlin")


def write(qh: DataFrame, daily: DataFrame, out: Path) -> dict[str, Path]:
    """15-minute table Hive-partitioned by year, daily table as one file."""
    paths = {"15min": out / "energy_15min", "daily": out / "energy_daily"}
    (
        qh.withColumn("year", F.year("date_berlin"))
        .repartition("year")
        .sortWithinPartitions("ts_utc")
        .write.mode("overwrite")
        .partitionBy("year")
        .parquet(str(paths["15min"]))
    )
    daily.coalesce(1).write.mode("overwrite").parquet(str(paths["daily"]))
    return paths
