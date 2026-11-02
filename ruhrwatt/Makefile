.PHONY: setup test test-fast ci serving test-rust lint fmt bench smard-prices smard-all bronze silver gold backtest pipeline lab spark spark-parity bench-spark rag-setup rag-index ask chat rag-eval rag-batch go-test api api-check docker-build up down pipeline-docker timer-install

# Needs uv (https://docs.astral.sh/uv/) and a Rust toolchain (https://rustup.rs).
setup:
	uv sync

test:
	uv run pytest

# Without the Spark tests (no Java needed, faster)
test-fast:
	uv run pytest -m 'not spark'

# Everything CI runs, locally
ci: lint test go-test

test-rust:
	cargo test -p ruhrwatt-core

lint:
	uv run ruff check . && uv run ruff format --check . && cargo fmt --all --check && cargo clippy -p ruhrwatt-core -- -D warnings

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

# JSON snapshots for the Go API (data/serving/)
serving:
	uv run ruhrwatt gold export-serving

# Full refresh: download new weeks, parse, join, gold, serving snapshot
pipeline: smard-all bronze silver gold serving

# JupyterLab in the project environment (notebooks/ reads data/silver)
lab:
	uv run jupyter lab --notebook-dir=notebooks

# Same silver + gold features with PySpark (Java 17/21) -> data/spark/
spark:
	uv run ruhrwatt spark build

# Spark and Polars must produce identical tables from the same bronze files
spark-parity:
	uv run ruhrwatt spark parity

# Polars vs Spark on synthetic data of growing size (YEARS="2 10 25")
YEARS ?= 2 10 25
bench-spark:
	uv run python benchmarks/bench_spark.py --years $(YEARS)

# --- local RAG assistant (Ollama) -------------------------------------------------------
LLM ?= qwen3:8b
EMBED ?= bge-m3

# Pull the chat and embedding models (~5 GB + ~1.2 GB), embed the docs, check everything
rag-setup:
	ollama pull $(LLM) && ollama pull $(EMBED) && uv run ruhrwatt rag index && uv run ruhrwatt rag doctor

# Embed the docs into data/rag/docs.lance (re-run after editing docs; unchanged chunks are reused)
rag-index:
	uv run ruhrwatt rag index

# make ask Q="Wie hoch war der Strompreis im September 2026?"
ask:
	uv run ruhrwatt rag ask "$(Q)" --verbose

chat:
	uv run ruhrwatt rag chat

# Evaluation set; compare models with: make rag-eval MODELS="qwen3:8b qwen3:14b"
MODELS ?= $(LLM)
rag-eval:
	uv run ruhrwatt rag eval $(foreach m,$(MODELS),--model $(m))

# All questions of a JSON file in one go -> data/rag/batch_<model>.md
# make rag-batch FILE=questions/fragen_de.json LLM=qwen3:14b
FILE ?= questions/fragen_de.json
rag-batch:
	uv run ruhrwatt rag batch $(FILE) --model $(LLM)

# --- Go API, Docker, scheduling -------------------------------------------------------
go-test:
	cd api && gofmt -l . && go vet ./... && go test -race ./...

# Run the API locally against data/serving: make api [PORT=8090]
PORT ?= 8080
api:
	cd api && ADDR=:$(PORT) RUHRWATT_SERVING_DIR=../data/serving go run .

# Quick check of a running API: make api-check [PORT=8090]
api-check:
	curl -fsS localhost:$(PORT)/readyz
	curl -fsS localhost:$(PORT)/v1/freshness; echo
	curl -fsS "localhost:$(PORT)/v1/cheapest/$$(date -d tomorrow +%F)?hours=4&contiguous=true"; echo

docker-build:
	docker compose --profile batch --profile spark build

up:
	docker compose up -d

down:
	docker compose --profile monitoring --profile assistant down

pipeline-docker:
	docker compose run --rm pipeline

# Pipeline at 14:15 and 19:30 Berlin via a systemd user timer
timer-install:
	mkdir -p ~/.config/systemd/user
	cp deploy/systemd/ruhrwatt-pipeline.* ~/.config/systemd/user/
	systemctl --user daemon-reload
	systemctl --user enable --now ruhrwatt-pipeline.timer
	systemctl --user list-timers ruhrwatt-pipeline.timer
