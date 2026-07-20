#!/usr/bin/env bash
#SBATCH --job-name=eiyw-prepare-safe
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
cd "${REPO_ROOT}"

: "${ENV_SCRIPT:?ENV_SCRIPT is required}"
: "${SAFE_MODELS:?SAFE_MODELS is required}"

source "${ENV_SCRIPT}"

echo "[prepare] repo=${REPO_ROOT}"
echo "[prepare] safe_models=${SAFE_MODELS}"
echo "[prepare] python=$(command -v python)"

mkdir -p "${REPO_ROOT}/logs"
chmod 755 "${REPO_ROOT}/logs" || true

echo "[prepare] start=$(date '+%Y-%m-%d %H:%M:%S %Z')"
echo "[prepare] slurm_job_id=${SLURM_JOB_ID:-unset}"
echo "[prepare] slurm_job_name=${SLURM_JOB_NAME:-unset}"
echo "[prepare] log_file=${REPO_ROOT}/logs/${SLURM_JOB_NAME:-job}-${SLURM_JOB_ID:-manual}.out"

make data

for model in ${SAFE_MODELS}; do
  echo "[prepare] downloading ${model}"
  make model MODEL="${model}"
done

echo "[prepare] done=$(date '+%Y-%m-%d %H:%M:%S %Z')"
