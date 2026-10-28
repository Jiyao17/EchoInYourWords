"""Compare the active Python to the captured experiment versions."""

import argparse
import importlib
import importlib.metadata
import platform
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-cuda", action="store_true")
    args = parser.parse_args()
    errors = []
    if platform.python_version() != "3.11.5":
        errors.append(f"Python: expected 3.11.5, got {platform.python_version()}")
    pins = Path(__file__).with_name("requirements-experiments.txt")
    for line in pins.read_text().splitlines():
        if not line or line.startswith("#"):
            continue
        name, expected = line.split("==", 1)
        try:
            actual = importlib.metadata.version(name)
            if name == "torch":
                actual = importlib.import_module("torch").__version__
            if actual != expected:
                errors.append(f"{name}: expected {expected}, got {actual}")
        except importlib.metadata.PackageNotFoundError:
            errors.append(f"{name}: missing")
        except Exception as exc:
            errors.append(f"{name}: {exc}")
    for module in ("torch", "transformers", "accelerate", "datasets"):
        try:
            importlib.import_module(module)
        except Exception as exc:
            errors.append(f"{module}: import failed: {exc}")
    try:
        torch = importlib.import_module("torch")
        print(f"PyTorch {torch.__version__}; CUDA {torch.version.cuda}; "
              f"cuDNN {torch.backends.cudnn.version()}")
        if torch.version.cuda != "12.6":
            errors.append("Expected PyTorch built with CUDA 12.6")
        if torch.backends.cudnn.version() != 90501:
            errors.append("Expected cuDNN 9.5.1")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if args.require_cuda:
            if not torch.cuda.is_available():
                errors.append("A working CUDA GPU is required")
            else:
                for i in range(torch.cuda.device_count()):
                    x = torch.ones((32, 32), device=f"cuda:{i}")
                    assert (x @ x).sum().item() == 32768
                    print(f"GPU {i}: {torch.cuda.get_device_name(i)}; matmul passed")
    except Exception as exc:
        errors.append(f"PyTorch runtime check failed: {exc}")
    for error in errors:
        print(f"ERROR: {error}")
    if errors:
        raise SystemExit(1)
    print("All captured experiment versions and requested runtime checks passed.")


if __name__ == "__main__":
    main()
