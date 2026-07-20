#!/usr/bin/env bash
#SBATCH --job-name=eiyw-sweep-safe
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
cd "${REPO_ROOT}"

: "${ENV_SCRIPT:?ENV_SCRIPT is required}"
: "${SAFE_MODELS:?SAFE_MODELS is required}"

EXP_STAGE="${EXP_STAGE:-smoke}"
CONCURRENCY_PER_GPU="${CONCURRENCY_PER_GPU:-1}"
RESUME="${RESUME:-0}"
METHOD="${METHOD:-}"

source "${ENV_SCRIPT}"

mkdir -p "${REPO_ROOT}/logs"
chmod 755 "${REPO_ROOT}/logs" || true

echo "[sweep] start=$(date '+%Y-%m-%d %H:%M:%S %Z')"
echo "[sweep] slurm_job_id=${SLURM_JOB_ID:-unset}"
echo "[sweep] slurm_job_name=${SLURM_JOB_NAME:-unset}"
echo "[sweep] log_file=${REPO_ROOT}/logs/${SLURM_JOB_NAME:-job}-${SLURM_JOB_ID:-manual}.out"
echo "[sweep] stage=${EXP_STAGE}"
echo "[sweep] safe_models=${SAFE_MODELS}"
echo "[sweep] method=${METHOD:-all}"
echo "[sweep] cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-unset}"
echo "[sweep] python=$(command -v python)"

CMD=(
  python scripts/run_all.py
  --stage "${EXP_STAGE}"
  --concurrency-per-gpu "${CONCURRENCY_PER_GPU}"
)

if [[ "${RESUME}" == "1" ]]; then
  CMD+=(--resume)
fi

if [[ -n "${METHOD}" ]]; then
  CMD+=(--method "${METHOD}")
fi

for model in ${SAFE_MODELS}; do
  CMD+=(--sharer "${model}")
  CMD+=(--receiver "${model}")
done

printf '[sweep] command:'
printf ' %q' "${CMD[@]}"
printf '\n'

"${CMD[@]}"

echo "[sweep] done=$(date '+%Y-%m-%d %H:%M:%S %Z')"
