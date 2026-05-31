#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/scratch/nt612/Github/Segment Macro Betas"
EXPECTED_JOB_ID="5752806"
RAW_RUN_ID="${RAW_RUN_ID:?RAW_RUN_ID required}"
RUN_ID="${1:-$(date -u +%Y%m%dT%H%M%SZ)}"
YEARS="${YEARS:-2006-2025}"
EXECUTE="${EXECUTE:-0}"
PYTHON_BIN="${PYTHON_BIN:-$HOME/.conda/envs/ml_core/bin/python}"

cd "$PROJECT_ROOT"
mkdir -p "runs/${RUN_ID}/logs" "runs/${RUN_ID}/manifests" "runs/${RUN_ID}/reports"

export PYTHONPATH="$PROJECT_ROOT/src:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

{
  echo "filing_dates_run_id=${RUN_ID}"
  echo "raw_run_id=${RAW_RUN_ID}"
  echo "years=${YEARS}"
  echo "execute=${EXECUTE}"
  echo "started_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "hostname=$(hostname)"
  echo "pwd=$(pwd)"
  echo "slurm_job_id=${SLURM_JOB_ID:-unset}"

  if [[ "${SLURM_JOB_ID:-}" != "${EXPECTED_JOB_ID}" ]]; then
    echo "ERROR: expected SLURM_JOB_ID=${EXPECTED_JOB_ID}, got ${SLURM_JOB_ID:-unset}" >&2
    exit 2
  fi

  args=(
    -m segment_macro_betas.filing_dates
    --project-root "$PROJECT_ROOT"
    --raw-run-id "$RAW_RUN_ID"
    --run-id "$RUN_ID"
    --years "$YEARS"
  )
  if [[ "$EXECUTE" == "1" ]]; then
    args+=(--execute)
  fi

  "$PYTHON_BIN" "${args[@]}"

  echo "finished_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
} > "runs/${RUN_ID}/logs/filing_dates_extract.out" 2> "runs/${RUN_ID}/logs/filing_dates_extract.err"

echo "$RUN_ID"
