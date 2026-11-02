"""Local SparkSession tuned for a single workstation."""

from __future__ import annotations

import os
import warnings

# PySpark warns about pandas 3 on import; this project moves data via Arrow, not pandas.
warnings.filterwarnings("ignore", message="PySpark does not yet fully support pandas")

from pyspark.sql import SparkSession  # noqa: E402

JAVA_HINT = (
    "Spark needs Java 17 or 21, but none was found (no JAVA_HOME, no `java` on PATH). "
    "Install it, e.g. `sudo apt install openjdk-21-jdk-headless`, then check `java -version`."
)


class JavaMissingError(RuntimeError):
    pass


def check_java() -> None:
    """Fail fast with an install hint instead of PySpark's JAVA_GATEWAY_EXITED traceback."""
    import shutil
    from pathlib import Path

    home = os.environ.get("JAVA_HOME")
    if home and (Path(home) / "bin" / "java").exists():
        return
    if shutil.which("java"):
        return
    raise JavaMissingError(JAVA_HINT)


def get_spark(app: str = "ruhrwatt") -> SparkSession:
    """Local session on all cores. Driver memory from RUHRWATT_SPARK_MEMORY (default 4g).

    Session time zone is UTC so timestamps are never shifted silently; Berlin time is
    always derived explicitly.
    """
    check_java()
    cores = os.environ.get("RUHRWATT_SPARK_CORES", "*")
    spark = (
        SparkSession.builder.appName(app)
        .master(f"local[{cores}]")
        .config("spark.driver.memory", os.environ.get("RUHRWATT_SPARK_MEMORY", "4g"))
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.sql.shuffle.partitions", os.environ.get("RUHRWATT_SPARK_PARTITIONS", "16"))
        .config("spark.sql.adaptive.enabled", "true")
        .config("spark.sql.execution.arrow.pyspark.enabled", "true")
        .config("spark.ui.showConsoleProgress", "false")
        .config("spark.ui.enabled", "false")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")
    return spark
