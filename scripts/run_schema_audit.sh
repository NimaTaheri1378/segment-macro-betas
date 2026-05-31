#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_amarel_env.sh"
RUN_ID="${1:-$(date -u +%Y%m%dT%H%M%SZ)}"

cd "$PROJECT_ROOT"

mkdir -p "runs/${RUN_ID}/logs" "runs/${RUN_ID}/manifests" "runs/${RUN_ID}/reports" configs

{
  echo "schema_audit_run_id=${RUN_ID}"
  echo "started_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "hostname=$(hostname)"
  echo "pwd=$(pwd)"
  echo "slurm_job_id=${SLURM_JOB_ID:-unset}"
  echo "slurm_job_name=${SLURM_JOB_NAME:-unset}"
  echo "slurm_job_gpus=${SLURM_JOB_GPUS:-unset}"

  if [[ "${SLURM_JOB_ID:-}" != "${EXPECTED_JOB_ID}" ]]; then
    echo "ERROR: expected SLURM_JOB_ID=${EXPECTED_JOB_ID}, got ${SLURM_JOB_ID:-unset}" >&2
    exit 2
  fi

  "$PYTHON_BIN" scripts/schema_audit.py --project-root "$PROJECT_ROOT" --run-id "$RUN_ID"
  echo "finished_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
} > "runs/${RUN_ID}/logs/schema_audit.out" 2> "runs/${RUN_ID}/logs/schema_audit.err"

echo "$RUN_ID"
