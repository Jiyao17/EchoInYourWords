# C2C Reproduction Notes

## Current status

Successfully ran the official C2C/Rosetta pretrained fuser pipeline on RunPod.

## Model pair

- Receiver/base model: Qwen/Qwen3-0.6B
- Sharer/teacher model: Qwen/Qwen2.5-0.5B-Instruct
- Method: Rosetta / C2C
- Fuser checkpoint: nics-efc/C2C_Fuser, qwen3_0.6b+qwen2.5_0.5b_Fuser/final

## Dataset

- MMLU-Redux
- Evaluation mode: generate
- Smoke test setting: sample_interval = 100

## Current smoke result

- Overall accuracy: 36.36%
- This is only a smoke test, not a full reproduction result.

## Notes

The current goal is to first reproduce C2C baseline results using the official pretrained fuser before implementing Echo.
