.PHONY: setup test test-rust lint fmt bench smard-prices smard-all bronze silver gold backtest pipeline lab

# Needs uv (https://docs.astral.sh/uv/) and a Rust toolchain (https://rustup.rs).
setup:
	uv sync

test:
	uv run pytest

test-rust:
	cargo test -p ruhrwatt-core

lint:
	uv run ruff check . && cargo clippy -p ruhrwatt-core -- -D warnings

fmt:
	uv run ruff format . && uv run ruff check --fix . && cargo fmt

# Rust core vs pure-Python reference (synthetic data; add --raw grid_load:DE for your own)
bench:
	uv run python benchmarks/bench_bronze.py

# Day-ahead prices since 2025-10-01 (first native 15-minute day)
smard-prices:
	uv run ruhrwatt smard download --dataset day_ahead_price --region DE-LU --since 2025-10-01

# All registered datasets for the default regions, since 2025-01-01
smard-all:
	uv run ruhrwatt smard download-all --since 2025-01-01

# Parse everything downloaded into bronze Parquet (Rust)
bronze:
	uv run ruhrwatt smard to-bronze-all

# Join bronze into silver tables (Polars) and run the quality gates
silver:
	uv run ruhrwatt silver build

# Forecasting features, load-shifting results, monthly mart
gold:
	uv run ruhrwatt gold build

# Day-ahead price forecast backtest (last 8 weeks held out)
backtest:
	uv run ruhrwatt gold backtest

# Full refresh: download new weeks, parse, join, gold
pipeline: smard-all bronze silver gold

# JupyterLab in the project environment (notebooks/ reads data/silver)
lab:
	uv run jupyter lab --notebook-dir=notebooks
