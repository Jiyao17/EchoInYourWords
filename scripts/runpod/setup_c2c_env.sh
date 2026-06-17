#!/usr/bin/env bash
set -e

echo "=== 0. Restore SSH public key ==="
mkdir -p /root/.ssh
chmod 700 /root/.ssh

cat > /root/.ssh/authorized_keys <<'EOF'
ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIHwo1IoYQlpV5BzjTvjycJ/tCxbbn57lfoX98yndwXEp Yaoyi.Jia@colorado.edu
EOF

chmod 600 /root/.ssh/authorized_keys

echo "=== 1. Check workspace ==="
ls -la /workspace

if [ ! -d "/workspace/C2C" ]; then
  echo "ERROR: /workspace/C2C not found"
  exit 1
fi

cd /workspace/C2C

echo "=== 2. Install C2C package and dependencies ==="
python -m pip install --upgrade pip
pip install -e .

echo "=== 3. Install missing runtime packages ==="
pip install huggingface_hub datasets accelerate jsonlines hf_transfer

echo "=== 4. Remove torch package conflicts ==="
pip uninstall -y torchvision torchaudio torchtext || true

echo "=== 5. Check Python packages ==="
python - <<'PY'
mods = ["torch", "transformers", "datasets", "accelerate", "jsonlines", "huggingface_hub"]
for m in mods:
    try:
        mod = __import__(m)
        print(m, "OK", getattr(mod, "__version__", ""))
    except Exception as e:
        print(m, "MISSING", e)
PY

echo "=== 6. Check C2C checkpoint ==="
if ls /workspace/.cache/huggingface/hub/models--nics-efc--C2C_Fuser/snapshots/*/qwen3_0.6b+qwen2.5_0.5b_Fuser/final >/dev/null 2>&1; then
  echo "checkpoint exists"
else
  echo "checkpoint missing, downloading..."
  python - <<'PY'
from huggingface_hub import snapshot_download
checkpoint_dir = snapshot_download(
    repo_id="nics-efc/C2C_Fuser",
    allow_patterns=["qwen3_0.6b+qwen2.5_0.5b_Fuser/*"],
)
print(checkpoint_dir)
PY
fi

echo "=== 7. Update C2C checkpoint path in configs ==="
python - <<'PY'
from pathlib import Path
import glob
import re

paths = glob.glob("/workspace/.cache/huggingface/hub/models--nics-efc--C2C_Fuser/snapshots/*/qwen3_0.6b+qwen2.5_0.5b_Fuser/final")
if not paths:
    print("WARNING: no checkpoint final path found")
else:
    final_path = paths[0]
    print("Using checkpoint:", final_path)

    config_files = [
        "/workspace/C2C/recipe/eval_recipe/mmlu_redux_c2c_smoke.yaml",
        "/workspace/C2C/recipe/eval_recipe/mmlu_redux_c2c_full.yaml",
    ]

    for f in config_files:
        p = Path(f)
        if not p.exists():
            print("skip missing", f)
            continue
        text = p.read_text()
        text = re.sub(r"checkpoints_dir: .+", f"checkpoints_dir: {final_path}", text)
        p.write_text(text)
        print("updated", f)
PY

echo "=== 8. Done ==="
echo "Now test with:"
echo "cd /workspace/C2C && python test_c2c_inference.py"
