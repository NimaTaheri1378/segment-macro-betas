#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_amarel_env.sh"
RUN_ID="${1:?run id required}"

cd "$PROJECT_ROOT"
mkdir -p "runs/${RUN_ID}/logs" "runs/${RUN_ID}/manifests" "artifacts/figures_static/${RUN_ID}"

export PYTHONPATH="$PROJECT_ROOT/src:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

{
  echo "smoke_figures_run_id=${RUN_ID}"
  echo "started_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "hostname=$(hostname)"
  echo "pwd=$(pwd)"
  echo "slurm_job_id=${SLURM_JOB_ID:-unset}"

  if [[ "${SLURM_JOB_ID:-}" != "${EXPECTED_JOB_ID}" ]]; then
    echo "ERROR: expected SLURM_JOB_ID=${EXPECTED_JOB_ID}, got ${SLURM_JOB_ID:-unset}" >&2
    exit 2
  fi

  "$PYTHON_BIN" -m segment_macro_betas.visualise.smoke_figures \
    --project-root "$PROJECT_ROOT" \
    --run-id "$RUN_ID"

  echo "finished_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
} > "runs/${RUN_ID}/logs/smoke_figures.out" 2> "runs/${RUN_ID}/logs/smoke_figures.err"
