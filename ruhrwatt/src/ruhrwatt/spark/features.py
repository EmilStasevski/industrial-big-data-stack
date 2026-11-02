"""Gold forecasting features in Spark, column for column the same as ``ruhrwatt.gold.features``.

Wall-clock lags are self-joins on the Berlin wall-clock timestamp shifted by k days; all
columns needed at one lag distance are joined at once (3 joins instead of 17). The repeated
autumn hour keeps its first occurrence via ``row_number`` over the wall-clock key.
"""

from __future__ import annotations

import math

from pyspark.sql import DataFrame, Window
from pyspark.sql import functions as F

from ruhrwatt.gold import features as G

WALL = "_wall"


def build_features(qh: DataFrame, daily: DataFrame) -> DataFrame:
    base = qh.select(
        "ts_utc",
        "ts_berlin",
        "date_berlin",
        F.col("ts_berlin").alias(WALL),
        G.TARGET,
        "price_is_native_15min",
        "hour_berlin",
        (F.col("hour_berlin") * 4 + F.floor(F.minute("ts_berlin") / 15))
        .cast("smallint")
        .alias("qh_of_day"),
        "weekday",
        F.month("ts_berlin").cast("tinyint").alias("month"),
        F.sin(2 * math.pi * F.dayofyear("ts_berlin") / 365.25).alias("doy_sin"),
        F.cos(2 * math.pi * F.dayofyear("ts_berlin") / 365.25).alias("doy_cos"),
        "is_workday",
        "is_holiday_nrw",
        "is_peak",
        *G.FORECASTS,
    )
    first = Window.partitionBy(WALL).orderBy("ts_utc")
    lookup = (
        qh.select(F.col("ts_berlin").alias(WALL), "ts_utc", G.TARGET, *G.FUNDAMENTALS)
        .withColumn("_n", F.row_number().over(first))
        .where(F.col("_n") == 1)
        .drop("_n", "ts_utc")
    )

    out = base
    for k in sorted(set(G.PRICE_LAG_DAYS) | set(G.FUNDAMENTAL_LAG_DAYS)):
        cols = []
        if k in G.PRICE_LAG_DAYS:
            cols.append(F.col(G.TARGET).alias(f"price_lag_{k}d"))
        if k in G.FUNDAMENTAL_LAG_DAYS:
            cols += [F.col(c).alias(f"{c.removesuffix('_mwh')}_lag_{k}d") for c in G.FUNDAMENTALS]
        shifted = lookup.select((F.col(WALL) + F.expr(f"INTERVAL {k} DAYS")).alias(WALL), *cols)
        out = out.join(shifted, WALL, "left")

    d1 = daily.select(
        F.date_add("date_berlin", 1).alias("date_berlin"),
        F.col("price_base").alias("price_d1_mean"),
        F.col("price_min").alias("price_d1_min"),
        F.col("price_max").alias("price_d1_max"),
        F.col("price_spread").alias("price_d1_spread"),
    )
    out = out.join(d1, "date_berlin", "left")
    fc = F.col("de_fc_wind_pv_mwh")
    day = Window.partitionBy("date_berlin")
    out = out.withColumns(
        {
            "de_fc_wind_pv_day_mean": F.avg(fc).over(day),
            "de_fc_wind_pv_day_max": F.max(fc).over(day),
            "de_fc_residual_proxy": F.col("de_load_lag_7d") - fc,
            "de_fc_wind_pv_share_proxy": fc / F.col("de_load_lag_7d"),
        }
    )
    head = ["ts_utc", "ts_berlin", "date_berlin", G.TARGET, "price_is_native_15min"]
    return out.select(*head, *G.feature_columns())
