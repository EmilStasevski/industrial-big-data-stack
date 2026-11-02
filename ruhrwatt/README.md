# Ruhrwatt

Small-big-data energy and logistics copilot for NRW industry: PySpark pipelines, Rust kernels
for the heavy parsing and joins, and a local RAG assistant that fits in 16 GB VRAM.

Part of `industrial-data-stack`.

## Quick start

Requires [uv](https://docs.astral.sh/uv/) and a Rust toolchain ([rustup](https://rustup.rs)):
`uv sync` compiles the Rust extension through maturin. `rust-toolchain.toml` pins the stable
channel, so rustup uses (or installs) stable Rust here even if your default is nightly.

```bash
uv sync                      # create .venv, build ruhrwatt._core, install everything
uv run pytest                # offline tests (Python + Rust parity), no network needed
uv run ruhrwatt smard list   # registered datasets
make smard-all               # download all series since 2025-01-01 into data/raw
make bronze                  # parse them with the Rust core into data/bronze (Parquet)
make silver                  # join into data/silver tables + quality gates (Polars)
make gold                    # forecasting features, load-shifting savings, monthly mart
make backtest                # walk-forward model comparison (ridge, HGB, LightGBM, XGBoost, CatBoost)
make pipeline                # download new weeks, bronze, silver, gold, serving snapshot
make lab                     # JupyterLab; start with notebooks/01_explore_silver.ipynb
make spark                   # silver + gold features with PySpark (Java 17/21) -> data/spark
make spark-parity            # proves Spark and Polars produce identical tables
make bench-spark             # Polars vs Spark on 2 / 10 / 25 years of synthetic data
make rag-setup && make rag-index   # local assistant: pull models (Ollama), embed the docs
make ask Q="What were the cheapest hours yesterday?"   # or: make chat / make rag-eval
make api                     # Go read-only API + metrics on :8080 (data/serving)
make up                      # same in Docker; see docs/devops.md for compose profiles
make timer-install           # pipeline at 14:15 and 19:30 Berlin (systemd user timer)
make bench                   # Rust vs pure-Python benchmark
```

PySpark is included (it needs Java 17 or 21 at run time). The assistant needs
[Ollama](https://ollama.com) with a 16 GB GPU or less (see `docs/rag.md`).

## Layout

```
src/ruhrwatt/
  cli.py              `ruhrwatt smard|silver|gold|spark|rag ...`
  download/smard.py   raw SMARD downloader (resumable, atomic writes, self-healing re-fetch)
  bronze/smard.py     raw -> bronze Parquet via the Rust core
  bronze/reference.py pure-Python oracle for the tests and the benchmark baseline
  silver/energy.py    bronze -> energy_15min + energy_daily (Polars, lazy)
  silver/gates.py     quality gates (price range, Amprion <= DE, PV dark at night, ...)
  gold/features.py    day-ahead forecasting features (no look-ahead, DST-safe lags)
  gold/backtest.py    walk-forward comparison of 8 models + calibrated P10/P90 ranges
  gold/shift.py       optimal within-day load shifting for NRW plant profiles
  gold/mart.py        gold tables incl. the monthly market mart
  gold/serving.py     JSON snapshots for the Go API
  spark/              PySpark silver + gold features, parity check vs Polars
  rag/                local assistant: Ollama client, data tools, docs index, agent, eval, batch
crates/ruhrwatt-core/ Rust (PyO3): parallel parse, dedupe, quality counters, Parquet writer
api/                  Go: read-only HTTP API over the serving snapshots, Prometheus metrics
benchmarks/           Rust vs Python parser, Polars vs Spark pipeline
notebooks/            01_explore_silver (market), 02_gold_forecast_and_shifting
questions/            question sets for `ruhrwatt rag batch` (fragen_de.json)
deploy/               systemd timer, Prometheus scrape config and alerts
scripts/pipeline.sh   the daily refresh (container and timer)
.github/              CI (Rust, Python, Spark, Go, Docker), Dependabot
data/raw/             untouched downloads, one folder per source
data/bronze/          parsed, typed Parquet (UTC timestamps), Hive-partitioned
data/silver/          cleaned and joined tables
data/gold/            features, backtest results and marts
data/serving/         JSON snapshots served by the Go API
docs/                 bronze/silver/gold schema, spark.md, rag.md, devops.md
tests/                pytest, offline (Go tests live in api/)
```

Data is never committed (see `.gitignore`); only the empty folder skeleton is.

## SMARD notes

- Bidding zone for prices is DE-LU. SMARD has no NRW-level data, so regional load and
  generation use the **Amprion** control area, which covers most of NRW.
- Always download `quarterhour`. Before 2025-10-01 it repeats each hourly price four times;
  from that date the day-ahead market is natively 15-minute. Bronze keeps this in the
  `is_native_15min` column.
- All 25 series are verified by a live download plus the silver gates (PV dark at night,
  Amprion <= Germany, TSO forecasts correlate 0.93-0.996 with their actuals).
  `ruhrwatt smard probe <filter ids>` checks new filter ids live before registering them.
- Weekly files downloaded before their week was final (SMARD fills weeks with a lag) are
  re-fetched automatically on the next run.
- The newest 2 weeks are re-fetched on every run because SMARD is still publishing them.
- A DST week has 668 (spring) or 676 (autumn) quarter-hours instead of 672; the parser
  keeps real UTC timestamps, so this needs no special handling.

Data: Bundesnetzagentur | SMARD.de, CC BY 4.0.
