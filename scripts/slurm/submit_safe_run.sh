#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SLURM_DIR="${REPO_ROOT}/scripts/slurm"

ENV_SCRIPT="${ENV_SCRIPT:-${SLURM_DIR}/hpg_env.sh}"
PARTITION="${PARTITION:-hpg-turin}"
ACCOUNT="${ACCOUNT:-}"
TIME_LIMIT="${TIME_LIMIT:-24:00:00}"
GRES="${GRES:-gpu:1}"
CPUS="${CPUS:-8}"
MEM="${MEM:-64G}"
METHOD="${METHOD:-}"
DATASET="${DATASET:-}"
EXP_STAGE="${EXP_STAGE:-smoke}"
CONCURRENCY_PER_GPU="${CONCURRENCY_PER_GPU:-1}"
RESUME="${RESUME:-0}"
INCLUDE_8B="${INCLUDE_8B:-0}"

SAFE_MODELS=(
  "unsloth/Llama-3.2-1B-Instruct"
  "unsloth/gemma-3-4b-it"
)
if [[ "${INCLUDE_8B}" == "1" || "${INCLUDE_8B}" == "true" || "${INCLUDE_8B}" == "TRUE" ]]; then
  SAFE_MODELS+=("mistralai/Ministral-8B-Instruct-2410")
fi
SAFE_MODELS_STR="${SAFE_MODELS[*]}"
LOG_DIR="${REPO_ROOT}/logs"

usage() {
  cat <<'EOF'
Usage:
  scripts/slurm/submit_safe_run.sh prepare
  scripts/slurm/submit_safe_run.sh method METHOD=lcfx DATASET=hotpotqa_bridge
  scripts/slurm/submit_safe_run.sh exp METHOD=t2t DATASET=ropes
  scripts/slurm/submit_safe_run.sh sweep

Environment variables:
  ENV_SCRIPT            default: scripts/slurm/hpg_env.sh
  PARTITION             default: hpg-turin
  ACCOUNT               default: unset
  TIME_LIMIT            default: 24:00:00
  GRES                  default: gpu:1
  CPUS                  default: 8
  MEM                   default: 64G
  INCLUDE_8B            default: 0
  EXP_STAGE             default: smoke
  CONCURRENCY_PER_GPU   default: 1
  RESUME                default: 0
  METHOD                required for method / exp
  DATASET               required for method / exp

Examples:
  scripts/slurm/submit_safe_run.sh prepare
  METHOD=t2t DATASET=hotpotqa_bridge scripts/slurm/submit_safe_run.sh exp
  EXP_STAGE=full RESUME=1 scripts/slurm/submit_safe_run.sh sweep
EOF
}

require_env_script() {
  if [[ ! -f "${ENV_SCRIPT}" ]]; then
    echo "Missing ENV_SCRIPT=${ENV_SCRIPT}" >&2
    echo "Copy scripts/slurm/hpg_env.sh.example to scripts/slurm/hpg_env.sh and edit it first." >&2
    exit 1
  fi
}

sbatch_common_args() {
  local args=()
  if [[ -n "${ACCOUNT}" ]]; then
    args+=("--account=${ACCOUNT}")
  fi
  args+=(
    "--partition=${PARTITION}"
    "--gres=${GRES}"
    "--cpus-per-task=${CPUS}"
    "--mem=${MEM}"
    "--time=${TIME_LIMIT}"
  )
  printf '%s\n' "${args[@]}"
}

submit() {
  local script_path="$1"
  shift
  local -a sbatch_args
  mapfile -t sbatch_args < <(sbatch_common_args)
  mkdir -p "${LOG_DIR}"
  echo "[submit] script=${script_path}"
  echo "[submit] env_script=${ENV_SCRIPT}"
  echo "[submit] safe_models=${SAFE_MODELS_STR}"
  echo "[submit] logs=${LOG_DIR}/%x-%j.out"
  echo "[submit] time=$(date '+%Y-%m-%d %H:%M:%S %Z')"
  echo "[submit] cwd=${REPO_ROOT}"
  echo "[submit] sbatch_args=${sbatch_args[*]}"
  echo "[submit] export=$*"
  sbatch "${sbatch_args[@]}" --export="$*" "${script_path}"
}

main() {
  local mode="${1:-}"
  if [[ -z "${mode}" || "${mode}" == "-h" || "${mode}" == "--help" ]]; then
    usage
    exit 0
  fi

  require_env_script

  case "${mode}" in
    prepare)
      submit "${SLURM_DIR}/run_safe_prepare.slurm" \
        "ALL,ENV_SCRIPT=${ENV_SCRIPT},SAFE_MODELS=${SAFE_MODELS_STR}"
      ;;
    method)
      if [[ -z "${METHOD}" || -z "${DATASET}" ]]; then
        echo "METHOD and DATASET are required for mode=method" >&2
        exit 1
      fi
      submit "${SLURM_DIR}/run_safe_method.slurm" \
        "ALL,ENV_SCRIPT=${ENV_SCRIPT},METHOD=${METHOD},DATASET=${DATASET},SAFE_MODELS=${SAFE_MODELS_STR}"
      ;;
    exp)
      if [[ -z "${METHOD}" || -z "${DATASET}" ]]; then
        echo "METHOD and DATASET are required for mode=exp" >&2
        exit 1
      fi
      submit "${SLURM_DIR}/run_safe_exp.slurm" \
        "ALL,ENV_SCRIPT=${ENV_SCRIPT},METHOD=${METHOD},DATASET=${DATASET},EXP_STAGE=${EXP_STAGE},CONCURRENCY_PER_GPU=${CONCURRENCY_PER_GPU},RESUME=${RESUME},SAFE_MODELS=${SAFE_MODELS_STR}"
      ;;
    sweep)
      submit "${SLURM_DIR}/run_safe_sweep.slurm" \
        "ALL,ENV_SCRIPT=${ENV_SCRIPT},EXP_STAGE=${EXP_STAGE},CONCURRENCY_PER_GPU=${CONCURRENCY_PER_GPU},RESUME=${RESUME},METHOD=${METHOD},SAFE_MODELS=${SAFE_MODELS_STR}"
      ;;
    *)
      echo "Unknown mode: ${mode}" >&2
      usage
      exit 1
      ;;
  esac
}

main "$@"
