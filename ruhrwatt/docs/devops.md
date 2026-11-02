# DevOps: CI, containers, serving, scheduling, monitoring

Each tool has one job; nothing here is required to run the pipeline locally with `make`.

| Piece | Job |
|---|---|
| GitHub Actions (`.github/workflows/ci.yml`) | every push: Rust fmt/clippy/tests, Python lint + tests, Spark parity on Java 21, Go vet/race tests, Docker builds, compose validation |
| pre-commit (`.pre-commit-config.yaml`) | ruff, cargo fmt, gofmt, YAML/TOML/JSON checks, blocks files > 1 MB (data never goes into git) |
| Dependabot (`.github/dependabot.yml`) | weekly updates for uv, cargo, Go, Actions and Docker base images |
| `Dockerfile` | pipeline image: Rust compiled in a builder stage, slim runtime, non-root; `spark` target adds Java |
| `api/` (Go) | read-only HTTP API over the gold data + Prometheus metrics; 6 MB static binary, distroless image |
| `docker-compose.yml` | API by default; pipeline, Spark, Ollama (GPU), assistant and Prometheus as profiles |
| `deploy/systemd/` | pipeline twice a day (14:15 and 19:30 Berlin): actuals, then tomorrow's prices once SMARD shows them |
| `deploy/prometheus/` | scrape config and alerts: API down, no data, data stale, tomorrow's prices missing |

## Git and CI

```bash
cd ~/industrial-big-data-stack/ruhrwatt
git init -b main && git add . && git commit -m "Ruhrwatt: SMARD pipeline, Rust parser, Spark, forecasting, local RAG"
git remote add origin git@github.com:<you>/ruhrwatt.git && git push -u origin main
uv run pre-commit install        # hooks on every commit
make ci                          # what CI runs, locally
```

`.gitignore` keeps `data/`, `.venv/`, `target/` and the Go binary out of the repository.

## Serving: batch writes, Go serves

The pipeline ends with `ruhrwatt gold export-serving`, which writes small JSON snapshots to
`data/serving/` (`monthly.json`, `prices.json` with the last 31 days of 15-minute prices,
`freshness.json` last). The Go service reads them and reloads within 30 s when
`freshness.json` changes, so a pipeline run needs no API restart. No Parquet or data
libraries in the service: the standard library only.

```bash
make api                                   # local, needs Go 1.23+; PORT=8090 if 8080 is taken
make api-check                             # readiness, freshness, tomorrow's cheapest block
curl localhost:8080/v1/monthly/2026-10
curl "localhost:8080/v1/cheapest/2026-11-02?hours=4&contiguous=true"
curl localhost:8080/v1/prices/2026-11-02
curl localhost:8080/metrics
```

Set `RUHRWATT_API_KEY` to require `X-API-Key: <key>` (or `Authorization: Bearer <key>`) on
`/v1/*`; health and metrics stay open. The cheapest-hours logic matches the assistant's
`cheapest_hours` tool (same block on the same day in a cross-check).

## Containers

```bash
make docker-build                  # pipeline + spark images (first build compiles Rust, ~5-10 min)
docker compose run --rm pipeline   # refresh data/ in a container
make up                            # API on :8080
docker compose --profile monitoring up -d    # + Prometheus on :9090 (Alerts tab)
```

The assistant with Ollama in a container needs the NVIDIA Container Toolkit
(`nvidia-ctk`). Since Ollama already runs on the host, the simpler route is:

```bash
OLLAMA_HOST=http://host.docker.internal:11434 docker compose --profile assistant run --rm assistant
```

On the host, Ollama listens on 127.0.0.1 by default; containers reach it only if it listens
on all interfaces (`OLLAMA_HOST=0.0.0.0` in the Ollama service settings).

## Daily schedule

```bash
make timer-install                 # systemd user timer, 14:15 + 19:30 Berlin, catches up after downtime
loginctl enable-linger $USER       # run even when you are not logged in
journalctl --user -u ruhrwatt-pipeline -n 50     # last run
systemctl --user start ruhrwatt-pipeline         # run now
```

The service runs `make pipeline` in the project folder (edit `WorkingDirectory` if the
project lives elsewhere); a commented line switches it to the Docker pipeline. A failing
quality gate makes the run fail, which `systemctl --user status` and the
`RuhrwattDataStale` alert both show.

## What was deliberately left out

Kubernetes, Terraform and a cloud registry: this is one workstation, and the point of the
project is small big data on one machine. The compose file and the static Go binary move to
a cluster unchanged if that ever matters.
