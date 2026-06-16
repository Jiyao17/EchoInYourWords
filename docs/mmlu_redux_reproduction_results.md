# MMLU-Redux Reproduction Results

## Setting

- Dataset: MMLU-Redux
- Evaluation mode: generate
- Full evaluation: sample_interval = 1
- Receiver model: Qwen/Qwen3-0.6B
- Sharer model for C2C: Qwen/Qwen2.5-0.5B-Instruct
- C2C checkpoint: nics-efc/C2C_Fuser, qwen3_0.6b+qwen2.5_0.5b_Fuser/final

## Results

| Method | Receiver | Sharer | Accuracy |
|---|---|---|---:|
| Receiver-only | Qwen/Qwen3-0.6B | N/A | 35.09% |
| C2C / Rosetta | Qwen/Qwen3-0.6B | Qwen/Qwen2.5-0.5B-Instruct | 43.08% |

## Improvement

C2C improves over Receiver-only by:

- Absolute improvement: +7.99 percentage points
- Relative improvement: approximately +22.8%

## Output files

- C2C summary: `results/Rosetta_mmlu-redux_generate_20260616_045643_summary.json`
- Receiver-only summary: `results/Qwen3-0.6B_mmlu-redux_generate_20260616_053359_summary.json`

## Notes

This result reproduces the core C2C comparison on MMLU-Redux: the same receiver model performs better when assisted by the C2C/Rosetta cache-fusion mechanism.
