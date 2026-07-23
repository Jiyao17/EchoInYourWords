#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SLURM_DIR="${REPO_ROOT}/scripts/slurm"

ENV_SCRIPT="${ENV_SCRIPT:-${SLURM_DIR}/hpg_env.sh}"
PARTITION="${PARTITION:-hpg-b200}"
ACCOUNT="${ACCOUNT:-}"
TIME_LIMIT="${TIME_LIMIT:-4-00:00:00}"
GRES="${GRES:-gpu:b200:1}"
CPUS="${CPUS:-16}"
MEM="${MEM:-128G}"
PREPARE_PARTITION="${PREPARE_PARTITION:-hpg-default}"
PREPARE_TIME_LIMIT="${PREPARE_TIME_LIMIT:-08:00:00}"
PREPARE_GRES="${PREPARE_GRES:-}"
PREPARE_CPUS="${PREPARE_CPUS:-4}"
PREPARE_MEM="${PREPARE_MEM:-32G}"
METHOD="${METHOD:-}"
DATASET="${DATASET:-}"
DATASETS="${DATASETS:-}"
EXP_STAGE="${EXP_STAGE:-smoke}"
CONCURRENCY_PER_GPU="${CONCURRENCY_PER_GPU:-1}"
RESUME="${RESUME:-0}"
DRY_RUN="${DRY_RUN:-0}"
PAIR_SCOPE="${PAIR_SCOPE:-all}"

SAFE_MODELS=(
  "unsloth/gemma-3-1b-it"
  "unsloth/Llama-3.2-3B-Instruct"
  "mistralai/Ministral-8B-Instruct-2410"
)
SAFE_MODELS_STR="${SAFE_MODELS[*]}"
LOG_DIR="${REPO_ROOT}/logs"
DEPENDENCY="${DEPENDENCY:-${SBATCH_DEPENDENCY:-}}"

usage() {
  cat <<'EOF'
Usage:
  scripts/slurm/submit_safe_run.sh prepare
  scripts/slurm/submit_safe_run.sh method METHOD=lcfx DATASET=hotpotqa_bridge
  scripts/slurm/submit_safe_run.sh exp METHOD=t2t DATASET=ropes
  scripts/slurm/submit_safe_run.sh sweep

Environment variables:
  ENV_SCRIPT            default: scripts/slurm/hpg_env.sh
  PARTITION             default: hpg-b200
  ACCOUNT               default: unset
  TIME_LIMIT            default: 4-00:00:00
  GRES                  default: gpu:b200:1
  CPUS                  default: 16
  MEM                   default: 128G
  EXP_STAGE             default: smoke
  PAIR_SCOPE            default: all
  CONCURRENCY_PER_GPU   default: 1
  RESUME                default: 0
  DRY_RUN               default: 0
  DEPENDENCY            optional sbatch dependency, e.g. afterok:<jobid>
  METHOD                required for method / exp
  DATASET               required for method / exp
  DATASETS              optional for sweep, space-separated dataset list
  PREPARE_PARTITION     default: hpg-default
  PREPARE_GRES          default: unset (CPU-only prepare)
  PREPARE_CPUS          default: 4
  PREPARE_MEM           default: 32G
  PREPARE_TIME_LIMIT    default: 08:00:00

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
    "--cpus-per-task=${CPUS}"
    "--mem=${MEM}"
    "--time=${TIME_LIMIT}"
  )
  if [[ -n "${GRES}" ]]; then
    args+=("--gres=${GRES}")
  fi
  if [[ -n "${DEPENDENCY}" ]]; then
    args+=("--dependency=${DEPENDENCY}")
  fi
  printf '%s\n' "${args[@]}"
}

submit() {
  local script_path="$1"
  shift
  local -a sbatch_args
  mapfile -t sbatch_args < <(sbatch_common_args)
  mkdir -p "${LOG_DIR}"
  chmod 755 "${LOG_DIR}" || true
  echo "[submit] script=${script_path}"
  echo "[submit] env_script=${ENV_SCRIPT}"
  echo "[submit] safe_models=${SAFE_MODELS_STR}"
  echo "[submit] logs=${LOG_DIR}/%x-%j.out"
  echo "[submit] time=$(date '+%Y-%m-%d %H:%M:%S %Z')"
  echo "[submit] cwd=${REPO_ROOT}"
  echo "[submit] sbatch_args=${sbatch_args[*]}"
  echo "[submit] export=$*"
  sbatch "${sbatch_args[@]}" --output="${LOG_DIR}/%x-%j.out" --export="$*" "${script_path}"
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
      PARTITION="${PREPARE_PARTITION}"
      TIME_LIMIT="${PREPARE_TIME_LIMIT}"
      GRES="${PREPARE_GRES}"
      CPUS="${PREPARE_CPUS}"
      MEM="${PREPARE_MEM}"
      submit "${SLURM_DIR}/run_safe_prepare.slurm" \
        "ALL,REPO_ROOT=${REPO_ROOT},ENV_SCRIPT=${ENV_SCRIPT},DATASETS=${DATASETS},SAFE_MODELS=${SAFE_MODELS_STR}"
      ;;
    method)
      if [[ -z "${METHOD}" || -z "${DATASET}" ]]; then
        echo "METHOD and DATASET are required for mode=method" >&2
        exit 1
      fi
      submit "${SLURM_DIR}/run_safe_method.slurm" \
        "ALL,REPO_ROOT=${REPO_ROOT},ENV_SCRIPT=${ENV_SCRIPT},METHOD=${METHOD},DATASET=${DATASET},SAFE_MODELS=${SAFE_MODELS_STR}"
      ;;
    exp)
      if [[ -z "${METHOD}" || -z "${DATASET}" ]]; then
        echo "METHOD and DATASET are required for mode=exp" >&2
        exit 1
      fi
      submit "${SLURM_DIR}/run_safe_exp.slurm" \
        "ALL,REPO_ROOT=${REPO_ROOT},ENV_SCRIPT=${ENV_SCRIPT},METHOD=${METHOD},DATASET=${DATASET},EXP_STAGE=${EXP_STAGE},PAIR_SCOPE=${PAIR_SCOPE},CONCURRENCY_PER_GPU=${CONCURRENCY_PER_GPU},RESUME=${RESUME},DRY_RUN=${DRY_RUN},SAFE_MODELS=${SAFE_MODELS_STR}"
      ;;
    sweep)
      submit "${SLURM_DIR}/run_safe_sweep.slurm" \
        "ALL,REPO_ROOT=${REPO_ROOT},ENV_SCRIPT=${ENV_SCRIPT},EXP_STAGE=${EXP_STAGE},PAIR_SCOPE=${PAIR_SCOPE},CONCURRENCY_PER_GPU=${CONCURRENCY_PER_GPU},RESUME=${RESUME},DRY_RUN=${DRY_RUN},METHOD=${METHOD},DATASETS=${DATASETS},SAFE_MODELS=${SAFE_MODELS_STR}"
      ;;
    *)
      echo "Unknown mode: ${mode}" >&2
      usage
      exit 1
      ;;
  esac
}

main "$@"
