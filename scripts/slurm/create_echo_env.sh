#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${ENV_NAME:-Echo}"
PYTHON_VERSION="${PYTHON_VERSION:-3.12}"

ensure_conda_module() {
  if command -v conda >/dev/null 2>&1; then
    return 0
  fi

  if command -v ml >/dev/null 2>&1; then
    ml conda >/dev/null 2>&1 || true
  fi

  if ! command -v conda >/dev/null 2>&1 && command -v module >/dev/null 2>&1; then
    module load conda >/dev/null 2>&1 || true
  fi
}

find_conda_sh() {
  if [[ -n "${CONDA_EXE:-}" ]]; then
    local base
    base="$(dirname "$(dirname "${CONDA_EXE}")")"
    if [[ -f "${base}/etc/profile.d/conda.sh" ]]; then
      echo "${base}/etc/profile.d/conda.sh"
      return 0
    fi
  fi

  local candidates=(
    "${HOME}/miniconda3/etc/profile.d/conda.sh"
    "${HOME}/anaconda3/etc/profile.d/conda.sh"
    "${HOME}/mambaforge/etc/profile.d/conda.sh"
    "${HOME}/miniforge3/etc/profile.d/conda.sh"
  )

  local path
  for path in "${candidates[@]}"; do
    if [[ -f "${path}" ]]; then
      echo "${path}"
      return 0
    fi
  done
  return 1
}

ensure_conda_module

CONDA_SH="$(find_conda_sh || true)"
if [[ -z "${CONDA_SH}" ]]; then
  echo "Could not find conda.sh automatically." >&2
  echo "On HPG, conda may require 'ml conda' before it becomes available." >&2
  echo "Set CONDA_EXE or edit this script with your conda installation path." >&2
  exit 1
fi

source "${CONDA_SH}"
eval "$(conda shell.bash hook)"

echo "[env] conda_sh=${CONDA_SH}"
echo "[env] conda=$(command -v conda)"
echo "[env] creating/updating env=${ENV_NAME} python=${PYTHON_VERSION}"

conda create -y -n "${ENV_NAME}" "python=${PYTHON_VERSION}"
conda activate "${ENV_NAME}"

python -m pip install --upgrade pip

# This repository does not currently ship a requirements.txt/environment.yml.
# The package set below is inferred from PROJECT_STRUCTURE.md and imports in
# scripts/ + methods/.
python -m pip install \
  --index-url https://download.pytorch.org/whl/cu118 \
  "torch==2.7.1+cu118" \
  "torchvision==0.22.1+cu118" \
  "torchaudio==2.7.1+cu118"

python -m pip install \
  transformers \
  datasets \
  huggingface_hub \
  accelerate \
  sentencepiece \
  safetensors

echo "[env] done"
python --version
python -c "import torch, transformers, datasets, huggingface_hub; print('torch', torch.__version__); print('cuda', torch.version.cuda); print('imports ok')"
