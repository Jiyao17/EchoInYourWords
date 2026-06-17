# RunPod Startup Guide

This document explains how to start a RunPod instance, connect it from VS Code, and initialize the C2C environment.

## 1. Start the RunPod instance

Start or resume the RunPod pod from the RunPod dashboard.

Use a pod with `/workspace` data migration enabled, so existing files can be reused after rebuilding:

```text
/workspace/C2C
/workspace/EchoInYourWords
/workspace/.cache
```

After the pod starts, open:

```text
RunPod Dashboard → Pods → target pod → Connect
```

Check the current SSH connection information:

```text
Host/IP
Port
User
```

The user is usually:

```text
root
```

## 2. Update local SSH config

On the local machine, edit the SSH config file.

Windows example:

```text
C:\Users\<LOCAL_USER>\.ssh\config
```

Add or update:

```sshconfig
Host runpod-vscode
    HostName <RUNPOD_IP>
    User root
    Port <RUNPOD_PORT>
    IdentityFile <PATH_TO_PRIVATE_KEY>
    IdentitiesOnly yes
```

Windows example:

```sshconfig
Host runpod-vscode
    HostName 195.26.xxx.xxx
    User root
    Port 28190
    IdentityFile C:/Users/<LOCAL_USER>/.ssh/id_ed25519
    IdentitiesOnly yes
```

## 3. Restore SSH public key on RunPod

After rebuilding or migrating a pod, SSH key login may not work immediately.

Open the RunPod Web Terminal first:

```text
RunPod Dashboard → Pods → target pod → Connect → Web Terminal
```

Run this in the Web Terminal:

```bash
mkdir -p /root/.ssh
chmod 700 /root/.ssh

cat > /root/.ssh/authorized_keys <<'EOF'
<YOUR_PUBLIC_SSH_KEY>
EOF

chmod 600 /root/.ssh/authorized_keys
```

Example public key format:

```text
ssh-ed25519 AAAA... user@example.com
```

Then test SSH from the local machine:

```bash
ssh runpod-vscode
```

If it logs into the pod without asking for a password, SSH is ready.

## 4. Connect from VS Code

Open VS Code locally.

Use:

```text
Remote-SSH: Connect to Host...
```

Select:

```text
runpod-vscode
```

After connecting, open:

```text
/workspace
```

For C2C work:

```text
/workspace/C2C
```

For this project repo:

```text
/workspace/EchoInYourWords
```

## 5. Initialize the C2C environment

After connecting to the pod, run:

```bash
bash /workspace/setup_c2c_env.sh
```

This script installs the C2C package, restores missing Python dependencies, checks the Hugging Face checkpoint, and updates local C2C config checkpoint paths.

## 6. Verify C2C works

Run:

```bash
cd /workspace/C2C
python test_c2c_inference.py
```

Expected output should include:

```text
C2C output text: Hello! How can I assist you today?
```

If this works, the C2C environment is ready.

## 7. Normal startup flow

Use this order after starting or rebuilding RunPod:

```text
1. Start RunPod pod
2. Check current IP and port
3. Update local SSH config
4. Open RunPod Web Terminal
5. Restore /root/.ssh/authorized_keys
6. SSH from local machine
7. Connect with VS Code Remote-SSH
8. Run /workspace/setup_c2c_env.sh
9. Verify with python test_c2c_inference.py
10. Continue experiments
```
