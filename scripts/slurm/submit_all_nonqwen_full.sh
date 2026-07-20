#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

PARTITION="${PARTITION:-hpg-turin}"
GRES="${GRES:-gpu:1}"
CPUS="${CPUS:-16}"
MEM="${MEM:-128G}"
TIME_LIMIT="${TIME_LIMIT:-48:00:00}"
CONCURRENCY_PER_GPU="${CONCURRENCY_PER_GPU:-1}"
INCLUDE_8B="${INCLUDE_8B:-1}"
RESUME="${RESUME:-1}"

methods=(t2t lcfx xkv)

echo "[batch-submit] root=${ROOT_DIR}"
echo "[batch-submit] partition=${PARTITION} gres=${GRES} cpus=${CPUS} mem=${MEM} time=${TIME_LIMIT}"
echo "[batch-submit] include_8b=${INCLUDE_8B} resume=${RESUME} concurrency_per_gpu=${CONCURRENCY_PER_GPU}"

for method in "${methods[@]}"; do
  echo "[batch-submit] submitting method=${method}"
  PARTITION="${PARTITION}" \
  GRES="${GRES}" \
  CPUS="${CPUS}" \
  MEM="${MEM}" \
  TIME_LIMIT="${TIME_LIMIT}" \
  METHOD="${method}" \
  EXP_STAGE=full \
  RESUME="${RESUME}" \
  INCLUDE_8B="${INCLUDE_8B}" \
  CONCURRENCY_PER_GPU="${CONCURRENCY_PER_GPU}" \
  bash scripts/slurm/submit_safe_run.sh sweep
done
