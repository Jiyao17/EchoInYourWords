# Environment handoff

Captured from the active **Anaconda base** environment on October 27, 2026.
This describes the interpreter used by plain `python` in this session; other
Conda environments were not exported. The repository revision is in `runtime.json`.

| Component | Observed version |
| --- | --- |
| OS / architecture | Ubuntu 22.04.4 / Linux x86_64, glibc 2.35 |
| Python | 3.11.5 |
| PyTorch | 2.7.1+cu126 (distribution metadata reports 2.7.1) |
| PyTorch CUDA runtime | 12.6 |
| cuDNN | 9.5.1 (package 9.5.1.17) |
| System CUDA toolkit / compiler | 12.1.1 / nvcc 12.1.105 |
| Transformers | 4.52.4 |
| Accelerate | 1.9.0 |
| Datasets | 3.3.2 |
| NumPy | 1.24.3 |
| Hugging Face Hub | 0.34.3 |
| Tokenizers / Safetensors | 0.21.4 / 0.5.3 |
| Triton | 3.3.1 |

The NVIDIA driver and GPU model could not be determined: `nvidia-smi` failed,
and PyTorch reported CUDA unavailable with zero devices in this execution session.
Do not interpret that result as proof that the underlying host has no GPUs.
The driver version in the toolkit's `version.json` describes the toolkit bundle,
not a detected running driver.

## Rebuild for experiments

Use a fresh environment on Linux x86_64 with an NVIDIA GPU and a working driver
compatible with the captured CUDA 12.6 PyTorch build. From the repository root:

```bash
conda create -n eiyw-experiments python=3.11.5 pip=25.1.1 -y
conda activate eiyw-experiments
python -m pip install 'torch==2.7.1+cu126' --index-url https://download.pytorch.org/whl/cu126
python -m pip install -r environment/requirements-experiments.txt
python -m pip check
python environment/verify.py --require-cuda
```

`requirements-experiments.txt` pins the installed versions of the experiment
dependencies and their recursive required dependencies, evaluated for the source
Python/platform. Seeds are torch, transformers, accelerate, datasets, modelscope
(the optional download fallback), sentencepiece, protobuf, pillow, pip, and wheel.
It includes 65 packages, including NVIDIA runtime packages and Triton. The captured
versions satisfy the package metadata requirements within this subset. The local
`python environment/verify.py` check passed, including the main library imports.
Installation into a clean
environment and GPU execution have **not** been tested here; this is an installed
version snapshot, not a wheel archive or a lock file with artifact hashes.

CUDA runtime libraries come from the pinned Python packages. The separate system
12.1 toolkit records what is installed here; it is not the runtime used to build
this PyTorch wheel. `nvcc` was absent from PATH, but the compiler at
`/usr/local/cuda-12.1/bin/nvcc` worked. The source shell references a nonexistent
CUDA 12.5 directory, and its `LD_LIBRARY_PATH` contains a literal shell expansion.
Set CUDA paths for the destination machine rather than copying those values.
If compiling custom CUDA extensions, separately provision a toolkit compatible
with the destination PyTorch build.

## Full original environment

- `packages.json` and `requirements-all.txt`: all 455 distributions visible to the
  active Python, with versions. The text file normalizes local build references
  into version pins; it is an inventory of the shared base environment.
- `pip-freeze-original.txt`: unmodified `pip freeze --all`, including Conda build
  paths that are unavailable on another machine.
- `conda-environment.yml`: full Conda export with build strings and its pip
  section; the source prefix is removed and the environment name is changed.
- `conda-explicit-linux-64.txt`: exact Conda artifact URLs for Linux x86_64.
  This contains Conda packages only; pip-installed packages require a separate
  install. `conda-packages.json` records Conda's view, including pip entries.
- `runtime.json`: interpreter, OS, CUDA/cuDNN/PyTorch build details and selected
  environment variables. `diagnostics.json` records commands, exit codes, and
  their output. `pip-check.txt` contains the source environment's conflicts.

For a best-effort full reconstruction, use
`conda env create -f environment/conda-environment.yml`. To reproduce just the
Conda artifacts, use
`conda create -n eiyw-conda-snapshot --file environment/conda-explicit-linux-64.txt`.
The base environment contains unrelated Anaconda applications and existing
missing/incompatible dependencies, so a full export may fail to solve or retain
those inconsistencies. Conda records can also lag behind pip replacements.
Keep the original snapshot as a reference; use the focused experiment setup above
to run this project's experiments. Artifact availability and platform differences
may require adjustments on the destination.

## Experiment inputs

Copy this directory together with the same repository revision to the destination,
or commit it before transferring the repository. These files do not include model
weights, datasets, checkpoints, or credentials. Use the project's `make prepare`
to download inputs, or copy `data/tasks` and `data/hf_cache` with symlink targets
preserved. Copy relevant `data/checkpoints` when resuming or evaluating a trained
translator. Use the same data/model revisions, explicit `SEED`, `DATA_SEED`, model
pair and experiment flags; software versions alone do not fix those inputs.

After validating the environment, follow the main README's Makefile commands,
setting `CUDA_VISIBLE_DEVICES` to the destination's GPU IDs. Differences in GPU
hardware and numerical kernels can still affect results.
