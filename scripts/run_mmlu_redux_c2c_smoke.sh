#!/usr/bin/env bash
set -e

cd /workspace/C2C

python script/evaluation/unified_evaluator.py \
  --config recipe/eval_recipe/mmlu_redux_c2c_smoke.yaml \
  2>&1 | tee logs_mmlu_redux_c2c_smoke.txt
