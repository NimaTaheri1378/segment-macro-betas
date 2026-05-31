#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_amarel_env.sh"
RAW_RUN_ID="${RAW_RUN_ID:?RAW_RUN_ID required}"
PANEL_RUN_ID="${PANEL_RUN_ID:?PANEL_RUN_ID required}"
MACRO_RUN_ID="${MACRO_RUN_ID:?MACRO_RUN_ID required}"
MACRO_DATASET="${MACRO_DATASET:-macro_official_monthly}"
RELEASE_LAG_DAYS="${RELEASE_LAG_DAYS:-0}"
RUN_ID="${1:-$(date -u +%Y%m%dT%H%M%SZ)}"

cd "$PROJECT_ROOT"
mkdir -p "runs/${RUN_ID}/logs" "runs/${RUN_ID}/manifests" "runs/${RUN_ID}/reports"

export PYTHONPATH="$PROJECT_ROOT/src:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

{
  echo "macro_tensor_run_id=${RUN_ID}"
  echo "raw_run_id=${RAW_RUN_ID}"
  echo "panel_run_id=${PANEL_RUN_ID}"
  echo "macro_run_id=${MACRO_RUN_ID}"
  echo "macro_dataset=${MACRO_DATASET}"
  echo "release_lag_days=${RELEASE_LAG_DAYS}"
  echo "started_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "hostname=$(hostname)"
  echo "pwd=$(pwd)"
  echo "slurm_job_id=${SLURM_JOB_ID:-unset}"

  if [[ "${SLURM_JOB_ID:-}" != "${EXPECTED_JOB_ID}" ]]; then
    echo "ERROR: expected SLURM_JOB_ID=${EXPECTED_JOB_ID}, got ${SLURM_JOB_ID:-unset}" >&2
    exit 2
  fi

  "$PYTHON_BIN" -m segment_macro_betas.macro_tensor \
    --project-root "$PROJECT_ROOT" \
    --run-id "$RUN_ID" \
    --raw-run-id "$RAW_RUN_ID" \
    --panel-run-id "$PANEL_RUN_ID" \
    --macro-run-id "$MACRO_RUN_ID" \
    --macro-dataset "$MACRO_DATASET" \
    --release-lag-days "$RELEASE_LAG_DAYS"

  echo "finished_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
} > "runs/${RUN_ID}/logs/macro_tensor.out" 2> "runs/${RUN_ID}/logs/macro_tensor.err"

echo "$RUN_ID"
