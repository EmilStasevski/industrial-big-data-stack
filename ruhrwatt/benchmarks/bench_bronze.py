"""Benchmark: Rust core vs pure-Python reference on the bronze step.

Both implementations do the same work (parse weekly JSON, sort, dedupe, quality counters,
write Parquet) and their outputs are checked for equality before any timing is reported.

    uv run python benchmarks/bench_bronze.py                 # synthetic, 520 weekly files
    uv run python benchmarks/bench_bronze.py --files 2000
    uv run python benchmarks/bench_bronze.py --raw grid_load:DE   # your downloaded data
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import tempfile
import time
from pathlib import Path

import pyarrow.parquet as pq

from ruhrwatt import _core
from ruhrwatt.bronze import smard as bronze
from ruhrwatt.bronze.reference import build_bronze_py

QH = 900_000
WEEK_MS = 7 * 24 * 3_600_000
START = 1_420_066_800_000  # 2015-01-01 00:00 Europe/Berlin, roughly: start of SMARD history


def make_synthetic(folder: Path, n_files: int) -> list[Path]:
    folder.mkdir(parents=True, exist_ok=True)
    files = []
    for w in range(n_files):
        t0 = START + w * WEEK_MS
        series = [
            [t0 + i * QH, None if i % 97 == 5 else round(40 + (i % 150) * 0.37, 2)]
            for i in range(672)
        ]
        path = folder / f"410_DE_quarterhour_{t0}.json"
        path.write_text(json.dumps({"meta_data": {}, "series": series}))
        files.append(path)
    return files


def best_of(fn, repeat: int) -> list[float]:
    times = []
    for _ in range(repeat):
        t = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t)
    return times


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--files", type=int, default=520, help="synthetic weekly files (520 ~ 10 y)")
    ap.add_argument("--repeat", type=int, default=5)
    ap.add_argument("--raw", metavar="DATASET:REGION", help="benchmark your downloaded raw files")
    args = ap.parse_args()

    with tempfile.TemporaryDirectory() as tmp_s:
        tmp = Path(tmp_s)
        if args.raw:
            dataset, region = args.raw.split(":")
            files = bronze.raw_files(dataset, region)
            if not files:
                raise SystemExit(f"no raw files for {args.raw}; run `ruhrwatt smard download`")
            label = f"{args.raw} (real data)"
        else:
            files = make_synthetic(tmp / "raw", args.files)
            label = "synthetic"
        size_mb = sum(f.stat().st_size for f in files) / 1e6
        paths = [str(f) for f in files]
        out_rs, out_py = tmp / "rs.parquet", tmp / "py.parquet"

        # Correctness first: identical output, or the timing means nothing.
        rs = _core.build_bronze(paths, str(out_rs), QH, None)
        py = build_bronze_py(files, out_py, QH, None)
        assert rs == py, f"stats differ: {rs} vs {py}"
        assert pq.read_table(out_rs).equals(pq.read_table(out_py)), "parquet output differs"

        t_rs = best_of(lambda: _core.build_bronze(paths, str(out_rs), QH, None), args.repeat)
        t_py = best_of(lambda: build_bronze_py(files, out_py, QH, None), max(2, args.repeat // 2))

    rows = rs["rows"]
    print(
        f"{label}: {len(files)} files, {size_mb:.1f} MB JSON, {rows:,} rows, {os.cpu_count()} CPUs"
    )
    print(f"{'impl':<14}{'median s':>10}{'best s':>10}{'rows/s':>14}")
    for name, t in (("python+pyarrow", t_py), ("rust (rayon)", t_rs)):
        print(f"{name:<14}{statistics.median(t):>10.3f}{min(t):>10.3f}{rows / min(t):>14,.0f}")
    print(f"speed-up (best vs best): {min(t_py) / min(t_rs):.1f}x")


if __name__ == "__main__":
    main()
