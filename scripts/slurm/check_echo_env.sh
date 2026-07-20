#!/usr/bin/env bash
set -euo pipefail

echo "[check] python=$(command -v python)"
python --version
python -c "import torch; print('torch', torch.__version__); print('torch.cuda', torch.version.cuda); print('cuda_available', torch.cuda.is_available())"
python -c "import transformers, datasets, huggingface_hub; print('imports ok')"
