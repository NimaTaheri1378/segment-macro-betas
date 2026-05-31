#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/scratch/nt612/Github/Segment Macro Betas"
EXPECTED_JOB_ID="5752806"
RUN_ID="${1:-$(date -u +%Y%m%dT%H%M%SZ)}"
SAMPLE_YEAR="${SMOKE_YEAR:-2019}"
MAX_FIRMS="${SMOKE_MAX_FIRMS:-40}"
MAX_SEGMENT_ROWS="${SMOKE_MAX_SEGMENT_ROWS:-50000}"
PYTHON_BIN="${PYTHON_BIN:-$HOME/.conda/envs/ml_core/bin/python}"

cd "$PROJECT_ROOT"
mkdir -p "runs/${RUN_ID}/logs" "runs/${RUN_ID}/manifests" "runs/${RUN_ID}/reports"

export PYTHONPATH="$PROJECT_ROOT/src:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

{
  echo "smoke_panel_run_id=${RUN_ID}"
  echo "started_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "hostname=$(hostname)"
  echo "pwd=$(pwd)"
  echo "slurm_job_id=${SLURM_JOB_ID:-unset}"
  echo "python=${PYTHON_BIN}"

  if [[ "${SLURM_JOB_ID:-}" != "${EXPECTED_JOB_ID}" ]]; then
    echo "ERROR: expected SLURM_JOB_ID=${EXPECTED_JOB_ID}, got ${SLURM_JOB_ID:-unset}" >&2
    exit 2
  fi

  "$PYTHON_BIN" -m segment_macro_betas.smoke_panel \
    --project-root "$PROJECT_ROOT" \
    --run-id "$RUN_ID" \
    --sample-year "$SAMPLE_YEAR" \
    --max-firms "$MAX_FIRMS" \
    --max-segment-rows "$MAX_SEGMENT_ROWS"

  echo "finished_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
} > "runs/${RUN_ID}/logs/smoke_panel.out" 2> "runs/${RUN_ID}/logs/smoke_panel.err"

echo "$RUN_ID"
