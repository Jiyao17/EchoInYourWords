#!/usr/bin/env bash
set -euo pipefail

ml conda
eval "$(conda shell.bash hook)"
conda activate Echo

python --version
which python
