#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/scratch/nt612/Github/Segment Macro Betas"
EXPECTED_JOB_ID="5752806"
RAW_RUN_ID="${RAW_RUN_ID:?RAW_RUN_ID required}"
PANEL_RUN_ID="${PANEL_RUN_ID:?PANEL_RUN_ID required}"
BASELINE_RUN_ID="${BASELINE_RUN_ID:?BASELINE_RUN_ID required}"
LGBM_RUN_ID="${LGBM_RUN_ID:?LGBM_RUN_ID required}"
SET_RUN_ID="${SET_RUN_ID:?SET_RUN_ID required}"
RUN_ID="${1:-$(date -u +%Y%m%dT%H%M%SZ)}"
PYTHON_BIN="${PYTHON_BIN:-$HOME/.conda/envs/ml_core/bin/python}"

cd "$PROJECT_ROOT"
mkdir -p "runs/${RUN_ID}/logs" "runs/${RUN_ID}/manifests" "runs/${RUN_ID}/reports"

export PYTHONPATH="$PROJECT_ROOT/src:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

{
  echo "visual_pack_run_id=${RUN_ID}"
  echo "raw_run_id=${RAW_RUN_ID}"
  echo "panel_run_id=${PANEL_RUN_ID}"
  echo "baseline_run_id=${BASELINE_RUN_ID}"
  echo "lgbm_run_id=${LGBM_RUN_ID}"
  echo "set_run_id=${SET_RUN_ID}"
  echo "started_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "hostname=$(hostname)"
  echo "pwd=$(pwd)"
  echo "slurm_job_id=${SLURM_JOB_ID:-unset}"

  if [[ "${SLURM_JOB_ID:-}" != "${EXPECTED_JOB_ID}" ]]; then
    echo "ERROR: expected SLURM_JOB_ID=${EXPECTED_JOB_ID}, got ${SLURM_JOB_ID:-unset}" >&2
    exit 2
  fi

  "$PYTHON_BIN" -m segment_macro_betas.visual_pack \
    --project-root "$PROJECT_ROOT" \
    --run-id "$RUN_ID" \
    --raw-run-id "$RAW_RUN_ID" \
    --panel-run-id "$PANEL_RUN_ID" \
    --baseline-run-id "$BASELINE_RUN_ID" \
    --lgbm-run-id "$LGBM_RUN_ID" \
    --set-run-id "$SET_RUN_ID"

  echo "finished_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
} > "runs/${RUN_ID}/logs/visual_pack.out" 2> "runs/${RUN_ID}/logs/visual_pack.err"

echo "$RUN_ID"
