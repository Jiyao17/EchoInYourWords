# RunPod Startup Notes

This note records what to do after starting, rebuilding, or migrating a RunPod instance.

## 1. Open RunPod Web Terminal first

After starting the pod, open:

```text
RunPod Dashboard → Pods → target pod → Connect → Web Terminal
```

Use the Web Terminal first because SSH key login may not work immediately after rebuilding or migrating the pod.

## 2. Restore SSH public key

Run the following commands in the RunPod Web Terminal:

```bash
mkdir -p /root/.ssh
chmod 700 /root/.ssh

cat > /root/.ssh/authorized_keys <<'EOF'
###your key###
EOF

chmod 600 /root/.ssh/authorized_keys
```

Then test SSH from local PowerShell:

```powershell
ssh runpod-vscode
```

If the pod IP or port changed, update the local SSH config first:

```text
C:\Users\jiaya\.ssh\config
```

Example:

```sshconfig
Host runpod-vscode
    HostName <NEW_RUNPOD_IP>
    User root
    Port <NEW_RUNPOD_PORT>
    IdentityFile C:/Users/jiaya/.ssh/id_ed25519
    IdentitiesOnly yes
```

## 3. Restore Python/C2C environment

After SSH works, run:

```bash
bash /workspace/setup_c2c_env.sh
```

This script installs the C2C package, restores missing Python dependencies, checks the Hugging Face checkpoint, and updates C2C config paths.

## 4. Verify C2C still works

Run:

```bash
cd /workspace/C2C
python test_c2c_inference.py
```

Expected output should contain something like:

```text
C2C output text: Hello! How can I assist you today?
```

If this works, the environment is ready.

## 5. Useful checks

Check workspace files:

```bash
ls -la /workspace
ls -la /workspace/C2C
ls -la /workspace/EchoInYourWords
```

Check C2C checkpoint:

```bash
ls /workspace/.cache/huggingface/hub/models--nics-efc--C2C_Fuser/snapshots/*/qwen3_0.6b+qwen2.5_0.5b_Fuser/final
```

Check installed Python packages:

```bash
python - <<'PY'
mods = ["torch", "transformers", "datasets", "accelerate", "jsonlines", "huggingface_hub"]
for m in mods:
    try:
        mod = __import__(m)
        print(m, "OK", getattr(mod, "__version__", ""))
    except Exception as e:
        print(m, "MISSING", e)
PY
```

## 6. Normal startup order

Use this order after every RunPod rebuild:

```text
1. Open Web Terminal
2. Restore SSH public key
3. SSH from local machine
4. Run /workspace/setup_c2c_env.sh
5. Run python test_c2c_inference.py
6. Continue experiments
```
