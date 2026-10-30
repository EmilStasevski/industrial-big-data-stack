"""Raw SMARD files -> bronze Parquet, using the Rust core.

Output layout (Hive-style, so Spark/Polars/DuckDB pick up the partitions as columns)::

    <data>/bronze/smard/dataset=<name>/region=<region>/resolution=<res>/part-0.parquet

Columns: ``ts_utc`` (timestamp[ms, UTC]), ``value`` (double, nullable),
``is_native_15min`` (bool; false only for the hourly prices repeated four times
before 2025-10-01).
"""

from __future__ import annotations

from pathlib import Path

from ruhrwatt import config
from ruhrwatt.download import smard as dl


def raw_files(
    dataset: str, region: str, resolution: str = "quarterhour", root: Path | None = None
) -> list[Path]:
    """Weekly raw files for one series, in timestamp order."""
    ds = dl.DATASETS[dataset]
    folder = dl.target_dir(root or config.raw_dir("smard"), ds.filter_id, region, resolution)
    return sorted(folder.glob(f"{ds.filter_id}_{region}_{resolution}_*.json"))


def bronze_path(
    dataset: str, region: str, resolution: str = "quarterhour", root: Path | None = None
) -> Path:
    base = root or (config.data_dir() / "bronze" / "smard")
    return (
        base
        / f"dataset={dataset}"
        / f"region={region}"
        / f"resolution={resolution}"
        / "part-0.parquet"
    )


def to_bronze(
    dataset: str,
    region: str,
    resolution: str = "quarterhour",
    raw_root: Path | None = None,
    bronze_root: Path | None = None,
) -> dict:
    """Convert one raw series to bronze Parquet. Returns the quality counters.

    Raises FileNotFoundError if nothing was downloaded yet, ValueError on malformed
    files or conflicting values for one timestamp.
    """
    from ruhrwatt import _core  # compiled Rust extension

    files = raw_files(dataset, region, resolution, raw_root)
    ds = dl.DATASETS[dataset]
    if not files:
        root = raw_root or config.raw_dir("smard")
        folder = dl.target_dir(root, ds.filter_id, region, resolution)
        raise FileNotFoundError(
            f"no raw files for {dataset}/{region}/{resolution} in {folder}; "
            "run `ruhrwatt smard download` first (or set RUHRWATT_DATA_DIR)"
        )
    out = bronze_path(dataset, region, resolution, bronze_root)
    return _core.build_bronze(
        [str(p) for p in files],
        str(out),
        dl.STEP_MS.get(resolution, 0),
        ds.native_from_ms,
    )
