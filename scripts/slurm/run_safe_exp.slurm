#!/usr/bin/env bash
#SBATCH --job-name=eiyw-exp-safe
#SBATCH --output=logs/%x-%j.out

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

: "${ENV_SCRIPT:?ENV_SCRIPT is required}"
: "${METHOD:?METHOD is required}"
: "${DATASET:?DATASET is required}"
: "${SAFE_MODELS:?SAFE_MODELS is required}"

EXP_STAGE="${EXP_STAGE:-smoke}"
CONCURRENCY_PER_GPU="${CONCURRENCY_PER_GPU:-1}"
RESUME="${RESUME:-0}"

source "${ENV_SCRIPT}"

mkdir -p logs

echo "[exp] start=$(date '+%Y-%m-%d %H:%M:%S %Z')"
echo "[exp] slurm_job_id=${SLURM_JOB_ID:-unset}"
echo "[exp] slurm_job_name=${SLURM_JOB_NAME:-unset}"
echo "[exp] log_file=logs/${SLURM_JOB_NAME:-job}-${SLURM_JOB_ID:-manual}.out"
echo "[exp] method=${METHOD} dataset=${DATASET} stage=${EXP_STAGE}"
echo "[exp] safe_models=${SAFE_MODELS}"
echo "[exp] cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-unset}"
echo "[exp] python=$(command -v python)"

CMD=(
  python scripts/run_sweep.py
  --dataset "${DATASET}"
  --method "${METHOD}"
  --stage "${EXP_STAGE}"
  --concurrency-per-gpu "${CONCURRENCY_PER_GPU}"
)

if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  CMD+=(--gpus "${CUDA_VISIBLE_DEVICES}")
fi

if [[ "${RESUME}" == "1" ]]; then
  CMD+=(--resume)
fi

for model in ${SAFE_MODELS}; do
  CMD+=(--sharer "${model}")
  CMD+=(--receiver "${model}")
done

printf '[exp] command:'
printf ' %q' "${CMD[@]}"
printf '\n'

"${CMD[@]}"

echo "[exp] done=$(date '+%Y-%m-%d %H:%M:%S %Z')"
