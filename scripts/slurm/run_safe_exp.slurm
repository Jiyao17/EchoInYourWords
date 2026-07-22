#!/usr/bin/env bash
#SBATCH --job-name=eiyw-exp-safe
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
cd "${REPO_ROOT}"

: "${ENV_SCRIPT:?ENV_SCRIPT is required}"
: "${METHOD:?METHOD is required}"
: "${DATASET:?DATASET is required}"
: "${SAFE_MODELS:?SAFE_MODELS is required}"

EXP_STAGE="${EXP_STAGE:-smoke}"
PAIR_SCOPE="${PAIR_SCOPE:-all}"
CONCURRENCY_PER_GPU="${CONCURRENCY_PER_GPU:-1}"
RESUME="${RESUME:-0}"
DRY_RUN="${DRY_RUN:-0}"

source "${ENV_SCRIPT}"

mkdir -p "${REPO_ROOT}/logs"
chmod 755 "${REPO_ROOT}/logs" || true

echo "[exp] start=$(date '+%Y-%m-%d %H:%M:%S %Z')"
echo "[exp] slurm_job_id=${SLURM_JOB_ID:-unset}"
echo "[exp] slurm_job_name=${SLURM_JOB_NAME:-unset}"
echo "[exp] log_file=${REPO_ROOT}/logs/${SLURM_JOB_NAME:-job}-${SLURM_JOB_ID:-manual}.out"
echo "[exp] method=${METHOD} dataset=${DATASET} stage=${EXP_STAGE}"
echo "[exp] pair_scope=${PAIR_SCOPE}"
echo "[exp] safe_models=${SAFE_MODELS}"
echo "[exp] cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-unset}"
echo "[exp] python=$(command -v python)"

CMD=(
  python scripts/run_sweep.py
  --dataset "${DATASET}"
  --method "${METHOD}"
  --stage "${EXP_STAGE}"
  --pair-scope "${PAIR_SCOPE}"
  --concurrency-per-gpu "${CONCURRENCY_PER_GPU}"
)

if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  CMD+=(--gpus "${CUDA_VISIBLE_DEVICES}")
fi

if [[ "${RESUME}" == "1" ]]; then
  CMD+=(--resume)
fi

if [[ "${DRY_RUN}" == "1" ]]; then
  CMD+=(--dry-run)
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
