#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_amarel_env.sh"
RUN_ID="${1:-$(date -u +%Y%m%dT%H%M%SZ)}"
START_DATE="${START_DATE:-2006-01-01}"
END_DATE="${END_DATE:-2025-12-31}"
EXECUTE="${EXECUTE:-0}"
SERIES_CONFIG="${SERIES_CONFIG:-configs/macro_series.yml}"
MACRO_REQUEST_DELAY_SECONDS="${MACRO_REQUEST_DELAY_SECONDS:-0}"

cd "$PROJECT_ROOT"
mkdir -p "runs/${RUN_ID}/logs" "runs/${RUN_ID}/manifests" "runs/${RUN_ID}/reports"

export PYTHONPATH="$PROJECT_ROOT/src:${PYTHONPATH:-}"

args=(--project-root "$PROJECT_ROOT" --run-id "$RUN_ID" --start "$START_DATE" --end "$END_DATE" --series-config "$SERIES_CONFIG" --request-delay-seconds "$MACRO_REQUEST_DELAY_SECONDS")
if [[ "$EXECUTE" == "1" ]]; then
  args+=(--execute)
fi

{
  echo "macro_engine_run_id=${RUN_ID}"
  echo "started_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "hostname=$(hostname)"
  echo "pwd=$(pwd)"
  echo "slurm_job_id=${SLURM_JOB_ID:-unset}"
  echo "execute=${EXECUTE}"
  echo "series_config=${SERIES_CONFIG}"
  echo "macro_request_delay_seconds=${MACRO_REQUEST_DELAY_SECONDS}"

  if [[ "${SLURM_JOB_ID:-}" != "${EXPECTED_JOB_ID}" ]]; then
    echo "ERROR: expected SLURM_JOB_ID=${EXPECTED_JOB_ID}, got ${SLURM_JOB_ID:-unset}" >&2
    exit 2
  fi

  "$PYTHON_BIN" -m segment_macro_betas.macro_engine "${args[@]}"
  echo "finished_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
} > "runs/${RUN_ID}/logs/macro_engine.out" 2> "runs/${RUN_ID}/logs/macro_engine.err"

echo "$RUN_ID"
