"""Run the Spark stages end to end, and check them against the Polars pipeline."""

from __future__ import annotations

import time
from pathlib import Path

import polars as pl

from ruhrwatt import config
from ruhrwatt.spark.session import get_spark


def spark_dir(root: Path | None = None) -> Path:
    return root or (config.data_dir() / "spark")


def to_polars(df) -> pl.DataFrame:
    """Spark -> Polars through Arrow (no pandas)."""
    return pl.from_arrow(df.toArrow())


def build(bronze_root: Path | None = None, out_root: Path | None = None) -> dict:
    """Silver (15-min + daily) and gold features with Spark, written as Parquet."""
    from pyspark.sql import functions as F

    from ruhrwatt.spark import features, silver

    t0 = time.perf_counter()
    spark = get_spark()
    t1 = time.perf_counter()
    qh = silver.build_15min(spark, bronze_root).cache()
    daily = silver.build_daily(qh).cache()
    out = spark_dir(out_root)
    paths = silver.write(qh, daily, out)
    feats = features.build_features(qh, daily)
    paths["features"] = out / "price_features_15min"
    (
        feats.withColumn("year", F.year("date_berlin"))
        .repartition("year")
        .sortWithinPartitions("ts_utc")
        .write.mode("overwrite")
        .partitionBy("year")
        .parquet(str(paths["features"]))
    )
    t2 = time.perf_counter()
    return {
        "qh": qh,
        "daily": daily,
        "paths": paths,
        "startup_s": t1 - t0,
        "run_s": t2 - t1,
        "rows": qh.count(),
    }


def parity(bronze_root: Path | None = None) -> dict[str, list]:
    """Build every table with both engines from the same bronze files and compare."""
    from ruhrwatt.gold import features as gold_features
    from ruhrwatt.silver import energy
    from ruhrwatt.spark import features, silver
    from ruhrwatt.spark.parity import compare

    spark = get_spark()
    s_qh = silver.build_15min(spark, bronze_root).cache()
    s_daily = silver.build_daily(s_qh).cache()
    s_feat = features.build_features(s_qh, s_daily)

    p_qh = energy.build_15min(bronze_root)
    p_daily = energy.build_daily(p_qh)
    p_feat = gold_features.build_features(p_qh, p_daily)
    return {
        "energy_15min": compare(p_qh, to_polars(s_qh)),
        "energy_daily": compare(p_daily, to_polars(s_daily), key="date_berlin"),
        "price_features_15min": compare(p_feat, to_polars(s_feat)),
    }
