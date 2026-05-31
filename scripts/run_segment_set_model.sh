#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_amarel_env.sh"
RAW_RUN_ID="${RAW_RUN_ID:?RAW_RUN_ID required}"
PANEL_RUN_ID="${PANEL_RUN_ID:?PANEL_RUN_ID required}"
RUN_ID="${1:-$(date -u +%Y%m%dT%H%M%SZ)}"
VARIANTS="${VARIANTS:-set_only,set_plus_controls}"
EPOCHS="${EPOCHS:-3}"
BATCH_SIZE="${BATCH_SIZE:-8192}"
MAX_SEGMENTS="${MAX_SEGMENTS:-12}"
MAX_VOCAB="${MAX_VOCAB:-512}"
MAX_TRAIN_ROWS_PER_FOLD="${MAX_TRAIN_ROWS_PER_FOLD:-}"
SET_DEVICE_TYPE="${SET_DEVICE_TYPE:-auto}"

cd "$PROJECT_ROOT"
mkdir -p "runs/${RUN_ID}/logs" "runs/${RUN_ID}/manifests" "runs/${RUN_ID}/reports"

export PYTHONPATH="$PROJECT_ROOT/src:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

{
  echo "segment_set_run_id=${RUN_ID}"
  echo "raw_run_id=${RAW_RUN_ID}"
  echo "panel_run_id=${PANEL_RUN_ID}"
  echo "variants=${VARIANTS}"
  echo "epochs=${EPOCHS}"
  echo "batch_size=${BATCH_SIZE}"
  echo "set_device_type=${SET_DEVICE_TYPE}"
  echo "started_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "hostname=$(hostname)"
  echo "pwd=$(pwd)"
  echo "slurm_job_id=${SLURM_JOB_ID:-unset}"

  if [[ "${SLURM_JOB_ID:-}" != "${EXPECTED_JOB_ID}" ]]; then
    echo "ERROR: expected SLURM_JOB_ID=${EXPECTED_JOB_ID}, got ${SLURM_JOB_ID:-unset}" >&2
    exit 2
  fi

  args=(
    -m segment_macro_betas.segment_set_model
    --project-root "$PROJECT_ROOT"
    --raw-run-id "$RAW_RUN_ID"
    --panel-run-id "$PANEL_RUN_ID"
    --run-id "$RUN_ID"
    --variants "$VARIANTS"
    --epochs "$EPOCHS"
    --batch-size "$BATCH_SIZE"
    --max-segments "$MAX_SEGMENTS"
    --max-vocab "$MAX_VOCAB"
    --device-type "$SET_DEVICE_TYPE"
  )
  if [[ -n "$MAX_TRAIN_ROWS_PER_FOLD" ]]; then
    args+=(--max-train-rows-per-fold "$MAX_TRAIN_ROWS_PER_FOLD")
  fi

  "$PYTHON_BIN" "${args[@]}"

  echo "finished_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
} > "runs/${RUN_ID}/logs/segment_set_model.out" 2> "runs/${RUN_ID}/logs/segment_set_model.err"

echo "$RUN_ID"
