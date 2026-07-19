#!/usr/bin/env bash
#SBATCH --job-name=eiyw-method-safe
#SBATCH --output=logs/%x-%j.out

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

: "${ENV_SCRIPT:?ENV_SCRIPT is required}"
: "${METHOD:?METHOD is required}"
: "${DATASET:?DATASET is required}"
: "${SAFE_MODELS:?SAFE_MODELS is required}"

source "${ENV_SCRIPT}"

mkdir -p logs

echo "[method] start=$(date '+%Y-%m-%d %H:%M:%S %Z')"
echo "[method] slurm_job_id=${SLURM_JOB_ID:-unset}"
echo "[method] slurm_job_name=${SLURM_JOB_NAME:-unset}"
echo "[method] log_file=logs/${SLURM_JOB_NAME:-job}-${SLURM_JOB_ID:-manual}.out"
echo "[method] method=${METHOD} dataset=${DATASET}"
echo "[method] safe_models=${SAFE_MODELS}"
echo "[method] cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-unset}"
echo "[method] python=$(command -v python)"

for sharer in ${SAFE_MODELS}; do
  for receiver in ${SAFE_MODELS}; do
    echo "[method] running ${METHOD}: ${sharer} -> ${receiver}"
    make "${METHOD}" DATASET="${DATASET}" SHARER="${sharer}" RECEIVER="${receiver}"
  done
done

echo "[method] done=$(date '+%Y-%m-%d %H:%M:%S %Z')"
