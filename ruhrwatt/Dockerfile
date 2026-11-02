# syntax=docker/dockerfile:1.7
# Ruhrwatt pipeline image. Multi-stage: the Rust extension is compiled once in the builder,
# the runtime image only gets the finished virtualenv (no Rust toolchain, no compilers).
#
#   docker build -t ruhrwatt .                      # pipeline + assistant (default target)
#   docker build -t ruhrwatt:spark --target spark . # + PySpark and a Java 21 runtime

ARG PYTHON=3.13

# ---------------------------------------------------------------- builder
FROM python:${PYTHON}-slim-bookworm AS builder
RUN apt-get update && apt-get install -y --no-install-recommends build-essential curl ca-certificates \
 && rm -rf /var/lib/apt/lists/*
# Stable Rust (rust-toolchain.toml pins the channel) and uv
RUN curl -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal --default-toolchain stable
ENV PATH=/root/.cargo/bin:$PATH
COPY --from=ghcr.io/astral-sh/uv:0.9 /uv /usr/local/bin/uv
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never
WORKDIR /app

# 1) dependencies only: cached until pyproject.toml / uv.lock change
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project --no-default-groups --group rag
# 2) the project itself (maturin compiles crates/ruhrwatt-core)
COPY Cargo.toml Cargo.lock rust-toolchain.toml README.md ./
COPY crates ./crates
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=cache,target=/app/target \
    uv sync --frozen --no-default-groups --group rag --no-editable

FROM builder AS builder-spark
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=cache,target=/app/target \
    uv sync --frozen --no-default-groups --group rag --group spark --no-editable

# ---------------------------------------------------------------- runtime
FROM python:${PYTHON}-slim-bookworm AS runtime
# libgomp: OpenMP for LightGBM / XGBoost / CatBoost
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
 && rm -rf /var/lib/apt/lists/* \
 && useradd --create-home --uid 1000 ruhrwatt
WORKDIR /app
COPY --from=builder --chown=ruhrwatt /app/.venv /app/.venv
COPY --chown=ruhrwatt README.md ./
COPY --chown=ruhrwatt docs ./docs
COPY --chown=ruhrwatt questions ./questions
COPY --chown=ruhrwatt scripts ./scripts
ENV PATH=/app/.venv/bin:$PATH \
    RUHRWATT_DATA_DIR=/data \
    PYTHONUNBUFFERED=1
USER ruhrwatt
VOLUME ["/data"]
ENTRYPOINT ["ruhrwatt"]
CMD ["--help"]

FROM runtime AS spark
USER root
RUN apt-get update && apt-get install -y --no-install-recommends openjdk-17-jre-headless \
 && rm -rf /var/lib/apt/lists/*
COPY --from=builder-spark --chown=ruhrwatt /app/.venv /app/.venv
# Debian bookworm ships Java 17 (supported by PySpark 4); `java` on PATH is all Spark needs.
ENV SPARK_LOCAL_IP=127.0.0.1
USER ruhrwatt
