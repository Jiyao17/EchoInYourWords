#!/usr/bin/env bash
#SBATCH --job-name=eiyw-prepare-safe
#SBATCH --output=logs/%x-%j.out

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

: "${ENV_SCRIPT:?ENV_SCRIPT is required}"
: "${SAFE_MODELS:?SAFE_MODELS is required}"

source "${ENV_SCRIPT}"

echo "[prepare] repo=${REPO_ROOT}"
echo "[prepare] safe_models=${SAFE_MODELS}"
echo "[prepare] python=$(command -v python)"

mkdir -p logs

echo "[prepare] start=$(date '+%Y-%m-%d %H:%M:%S %Z')"
echo "[prepare] slurm_job_id=${SLURM_JOB_ID:-unset}"
echo "[prepare] slurm_job_name=${SLURM_JOB_NAME:-unset}"
echo "[prepare] log_file=logs/${SLURM_JOB_NAME:-job}-${SLURM_JOB_ID:-manual}.out"

make data

for model in ${SAFE_MODELS}; do
  echo "[prepare] downloading ${model}"
  make model MODEL="${model}"
done

echo "[prepare] done=$(date '+%Y-%m-%d %H:%M:%S %Z')"
