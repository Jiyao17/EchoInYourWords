# HPG Slurm Wrappers

These files are a thin HPG layer for this repository.

They do **not** change the repository's existing experiment code. Instead, they:

1. source your HPG Python environment
2. call the existing `make` / `python scripts/...` entrypoints
3. skip all Qwen models by default

They also pass the repository root explicitly into `sbatch`, so the Slurm copy
of the script running under `/var/spool/...` still writes logs and outputs back
to the real checkout instead of treating `/var/spool` as the project root.

## Files

- `submit_safe_run.sh`: preferred single entrypoint for Slurm submission
- `hpg_env.sh.example`: template for your HPG environment activation
- `run_safe_prepare.slurm`: prepare datasets and download only safe models
- `run_safe_method.slurm`: run one explicit method on one dataset over safe model pairs
- `run_safe_exp.slurm`: run one sweep cell family on one dataset with safe models only
- `run_safe_sweep.slurm`: run the multi-dataset sweep with safe models only
- `../../Makefile.hpg`: optional submit wrapper if you prefer `make`

## Default safe models

By default, all wrappers exclude Qwen and use only:

- `unsloth/Llama-3.2-1B-Instruct`
- `unsloth/gemma-3-4b-it`

If you want to also include the non-Qwen 8B model, set `INCLUDE_8B=1` when using
`Makefile.hpg`. That adds:

- `mistralai/Ministral-8B-Instruct-2410`

## First-time setup

0. Create a fresh `Echo` environment.

This repository does not currently ship a `requirements.txt` or `environment.yml`.
Use the helper scripts to create and check a clean conda env from the
dependencies inferred from the codebase and [PROJECT_STRUCTURE.md](../../PROJECT_STRUCTURE.md).

Important HPG note: `conda` may not exist in a fresh shell by default. On HPG,
load it first with `ml conda`. After that, prefer:

```bash
eval "$(conda shell.bash hook)"
conda activate Echo
```

instead of hardcoding a `.../etc/profile.d/conda.sh` path, because the active
HPG conda installation may not live under a fixed home-directory prefix.

```bash
bash scripts/slurm/create_echo_env.sh
bash scripts/slurm/check_echo_env.sh
```

1. Copy `hpg_env.sh.example` to `hpg_env.sh`
2. Edit `hpg_env.sh` so it loads `conda` with `ml conda` and activates `Echo`
3. Submit prepare first:

```bash
bash scripts/slurm/submit_safe_run.sh prepare
```

## Suggested run order

### 1. Prepare data and safe models

```bash
bash scripts/slurm/submit_safe_run.sh prepare
```

### 2. Smoke test one dataset

```bash
METHOD=t2t DATASET=hotpotqa_bridge bash scripts/slurm/submit_safe_run.sh exp
METHOD=lcfx DATASET=hotpotqa_bridge bash scripts/slurm/submit_safe_run.sh exp
METHOD=xkv DATASET=hotpotqa_bridge bash scripts/slurm/submit_safe_run.sh exp
```

### 3. Full run one dataset

```bash
METHOD=t2t DATASET=hotpotqa_bridge EXP_STAGE=full bash scripts/slurm/submit_safe_run.sh exp
```

### 4. Full multi-dataset sweep

```bash
EXP_STAGE=full bash scripts/slurm/submit_safe_run.sh sweep
```

## Preferred usage

If you just want one command surface, use `submit_safe_run.sh`.

- `prepare`: datasets + safe model downloads
- `method`: one method target over all safe model pairs on one dataset
- `exp`: one sweep mode on one dataset
- `sweep`: multi-dataset sweep

Examples:

```bash
bash scripts/slurm/submit_safe_run.sh prepare
METHOD=lcfx DATASET=ropes bash scripts/slurm/submit_safe_run.sh method
METHOD=t2t DATASET=musique EXP_STAGE=full bash scripts/slurm/submit_safe_run.sh exp
EXP_STAGE=full RESUME=1 bash scripts/slurm/submit_safe_run.sh sweep
INCLUDE_8B=1 EXP_STAGE=full bash scripts/slurm/submit_safe_run.sh sweep
```

## Why not use the repository defaults directly?

The repository defaults are excellent for a normal machine, but they are not
safe for your HPG policy because:

- bare `make model` downloads Qwen by default
- bare `make prepare` includes that model download
- bare `make exp` uses the default roster, which includes Qwen
- bare `make sweep` also reaches the Qwen-containing roster

These wrappers solve that by always passing explicit non-Qwen model filters.

## Results and upload

Current `.gitignore` still contains `/results/*`, so if you later need git to
track run outputs, remove or narrow that ignore rule before committing results.
