#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

PARTITION="${PARTITION:-hpg-b200}"
GRES="${GRES:-gpu:b200:1}"
CPUS="${CPUS:-16}"
MEM="${MEM:-128G}"
TIME_LIMIT="${TIME_LIMIT:-4-00:00:00}"
CONCURRENCY_PER_GPU="${CONCURRENCY_PER_GPU:-1}"
RESUME="${RESUME:-1}"
PAIR_SCOPE="${PAIR_SCOPE:-all}"

methods=(t2t lcfx xkv)
datasets=(musique 2wikimultihop)

echo "[batch-submit] root=${ROOT_DIR}"
echo "[batch-submit] partition=${PARTITION} gres=${GRES} cpus=${CPUS} mem=${MEM} time=${TIME_LIMIT}"
echo "[batch-submit] pair_scope=${PAIR_SCOPE} resume=${RESUME} concurrency_per_gpu=${CONCURRENCY_PER_GPU}"

for method in "${methods[@]}"; do
  for dataset in "${datasets[@]}"; do
    echo "[batch-submit] submitting method=${method} dataset=${dataset}"
    PARTITION="${PARTITION}" \
    GRES="${GRES}" \
    CPUS="${CPUS}" \
    MEM="${MEM}" \
    TIME_LIMIT="${TIME_LIMIT}" \
    METHOD="${method}" \
    DATASET="${dataset}" \
    EXP_STAGE=full \
    PAIR_SCOPE="${PAIR_SCOPE}" \
    RESUME="${RESUME}" \
    CONCURRENCY_PER_GPU="${CONCURRENCY_PER_GPU}" \
    bash scripts/slurm/submit_safe_run.sh exp
  done
done
