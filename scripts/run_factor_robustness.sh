#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_amarel_env.sh"
PANEL_RUN_ID="${PANEL_RUN_ID:?PANEL_RUN_ID required}"
MODEL_RUNS="${MODEL_RUNS:?MODEL_RUNS required, e.g. lgbm:run_id,deepsets:run_id}"
COST_BPS="${COST_BPS:-10}"
NW_LAG="${NW_LAG:-6}"
RUN_ID="${1:-$(date -u +%Y%m%dT%H%M%SZ)}"

cd "$PROJECT_ROOT"
mkdir -p "runs/${RUN_ID}/logs" "runs/${RUN_ID}/manifests" "runs/${RUN_ID}/reports"

export PYTHONPATH="$PROJECT_ROOT/src:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

{
  echo "factor_robustness_run_id=${RUN_ID}"
  echo "panel_run_id=${PANEL_RUN_ID}"
  echo "model_runs=${MODEL_RUNS}"
  echo "cost_bps=${COST_BPS}"
  echo "nw_lag=${NW_LAG}"
  echo "started_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "hostname=$(hostname)"
  echo "pwd=$(pwd)"
  echo "slurm_job_id=${SLURM_JOB_ID:-unset}"

  if [[ "${SLURM_JOB_ID:-}" != "${EXPECTED_JOB_ID}" ]]; then
    echo "ERROR: expected SLURM_JOB_ID=${EXPECTED_JOB_ID}, got ${SLURM_JOB_ID:-unset}" >&2
    exit 2
  fi

  "$PYTHON_BIN" -m segment_macro_betas.factor_robustness \
    --project-root "$PROJECT_ROOT" \
    --panel-run-id "$PANEL_RUN_ID" \
    --run-id "$RUN_ID" \
    --model-runs "$MODEL_RUNS" \
    --cost-bps "$COST_BPS" \
    --nw-lag "$NW_LAG"

  echo "finished_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
} > "runs/${RUN_ID}/logs/factor_robustness.out" 2> "runs/${RUN_ID}/logs/factor_robustness.err"

echo "$RUN_ID"
