#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_amarel_env.sh"
PANEL_RUN_ID="${PANEL_RUN_ID:?PANEL_RUN_ID required}"
LGBM_RUN_ID="${LGBM_RUN_ID:?LGBM_RUN_ID required}"
SET_RUN_ID="${SET_RUN_ID:?SET_RUN_ID required}"
FACTOR_RUN_ID="${FACTOR_RUN_ID:?FACTOR_RUN_ID required}"
RUN_ID="${1:-$(date -u +%Y%m%dT%H%M%SZ)}"

cd "$PROJECT_ROOT"
mkdir -p "runs/${RUN_ID}/logs" "runs/${RUN_ID}/manifests" "runs/${RUN_ID}/reports"

export PYTHONPATH="$PROJECT_ROOT/src:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

{
  echo "claim_ledger_run_id=${RUN_ID}"
  echo "panel_run_id=${PANEL_RUN_ID}"
  echo "lgbm_run_id=${LGBM_RUN_ID}"
  echo "set_run_id=${SET_RUN_ID}"
  echo "factor_run_id=${FACTOR_RUN_ID}"
  echo "started_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "hostname=$(hostname)"
  echo "pwd=$(pwd)"
  echo "slurm_job_id=${SLURM_JOB_ID:-unset}"

  if [[ "${SLURM_JOB_ID:-}" != "${EXPECTED_JOB_ID}" ]]; then
    echo "ERROR: expected SLURM_JOB_ID=${EXPECTED_JOB_ID}, got ${SLURM_JOB_ID:-unset}" >&2
    exit 2
  fi

  "$PYTHON_BIN" -m segment_macro_betas.claim_ledger \
    --project-root "$PROJECT_ROOT" \
    --run-id "$RUN_ID" \
    --panel-run-id "$PANEL_RUN_ID" \
    --lgbm-run-id "$LGBM_RUN_ID" \
    --set-run-id "$SET_RUN_ID" \
    --factor-run-id "$FACTOR_RUN_ID"

  echo "finished_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
} > "runs/${RUN_ID}/logs/claim_ledger.out" 2> "runs/${RUN_ID}/logs/claim_ledger.err"

echo "$RUN_ID"
