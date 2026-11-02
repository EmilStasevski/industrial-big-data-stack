#!/usr/bin/env bash
# Daily refresh: new SMARD weeks -> bronze -> silver (+ gates) -> gold -> serving snapshot.
# Used by the container (`docker compose run --rm pipeline`) and the systemd timer.
# Exits non-zero if a step or an error-level quality gate fails, so the timer shows it.
set -euo pipefail
SINCE="${RUHRWATT_SINCE:-2025-01-01}"

step() { echo "==> $*"; }
step "download (since ${SINCE})"; ruhrwatt smard download-all --since "${SINCE}"
step "bronze";                    ruhrwatt smard to-bronze-all
step "silver + quality gates";    ruhrwatt silver build
step "gold";                      ruhrwatt gold build
step "serving snapshot";          ruhrwatt gold export-serving
if [[ "${RUHRWATT_BACKTEST:-0}" == "1" ]]; then
  step "backtest";                ruhrwatt gold backtest
fi
step "done"
