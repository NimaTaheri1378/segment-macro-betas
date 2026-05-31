#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_amarel_env.sh"
PANEL_RUN_ID="${PANEL_RUN_ID:?PANEL_RUN_ID required}"
RUN_ID="${1:-$(date -u +%Y%m%dT%H%M%SZ)}"
N_JOBS="${N_JOBS:-${SLURM_CPUS_PER_TASK:-4}}"
MIN_TRAIN_MONTHS="${MIN_TRAIN_MONTHS:-36}"
MAX_TRAIN_ROWS_PER_FOLD="${MAX_TRAIN_ROWS_PER_FOLD:-}"
VARIANTS="${VARIANTS:-all,no_market_factors,no_return_or_market,segment_only,non_segment_controls}"
LGBM_DEVICE_TYPE="${LGBM_DEVICE_TYPE:-auto}"

cd "$PROJECT_ROOT"
mkdir -p "runs/${RUN_ID}/logs" "runs/${RUN_ID}/manifests" "runs/${RUN_ID}/reports"

export PYTHONPATH="$PROJECT_ROOT/src:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

{
  echo "lgbm_benchmark_run_id=${RUN_ID}"
  echo "panel_run_id=${PANEL_RUN_ID}"
  echo "started_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "hostname=$(hostname)"
  echo "pwd=$(pwd)"
  echo "slurm_job_id=${SLURM_JOB_ID:-unset}"
  echo "n_jobs=${N_JOBS}"
  echo "variants=${VARIANTS}"
  echo "lgbm_device_type=${LGBM_DEVICE_TYPE}"

  if [[ "${SLURM_JOB_ID:-}" != "${EXPECTED_JOB_ID}" ]]; then
    echo "ERROR: expected SLURM_JOB_ID=${EXPECTED_JOB_ID}, got ${SLURM_JOB_ID:-unset}" >&2
    exit 2
  fi

  args=(
    -m segment_macro_betas.lgbm_benchmark
    --project-root "$PROJECT_ROOT"
    --panel-run-id "$PANEL_RUN_ID"
    --run-id "$RUN_ID"
    --n-jobs "$N_JOBS"
    --min-train-months "$MIN_TRAIN_MONTHS"
    --variants "$VARIANTS"
    --device-type "$LGBM_DEVICE_TYPE"
  )
  if [[ -n "$MAX_TRAIN_ROWS_PER_FOLD" ]]; then
    args+=(--max-train-rows-per-fold "$MAX_TRAIN_ROWS_PER_FOLD")
  fi

  "$PYTHON_BIN" "${args[@]}"

  echo "finished_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
} > "runs/${RUN_ID}/logs/lgbm_benchmark.out" 2> "runs/${RUN_ID}/logs/lgbm_benchmark.err"

echo "$RUN_ID"
