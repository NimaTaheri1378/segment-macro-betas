#!/usr/bin/env bash
set -euo pipefail

: "${SMB_PROJECT_ROOT:?Set SMB_PROJECT_ROOT to the approved Amarel project root.}"
: "${SMB_SLURM_JOB_ID:?Set SMB_SLURM_JOB_ID to the approved Slurm allocation id.}"

PROJECT_ROOT="$SMB_PROJECT_ROOT"
EXPECTED_JOB_ID="$SMB_SLURM_JOB_ID"
PYTHON_BIN="${PYTHON_BIN:-$HOME/.conda/envs/ml_core/bin/python}"
