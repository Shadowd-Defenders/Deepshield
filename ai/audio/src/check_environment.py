"""Report the local audio AI development environment."""

from __future__ import annotations

import importlib.metadata
import platform
import sys


def package_version(package_name: str) -> str:
    try:
        return importlib.metadata.version(package_name)
    except importlib.metadata.PackageNotFoundError:
        return "not installed"


def print_cpu_information() -> None:
    print("CPU information:")
    print(f"  platform: {platform.platform()}")
    print(f"  machine: {platform.machine()}")
    print(f"  processor: {platform.processor() or 'unknown'}")

    try:
        import psutil

        print(f"  physical cores: {psutil.cpu_count(logical=False)}")
        print(f"  logical cores: {psutil.cpu_count(logical=True)}")
    except ImportError:
        print(f"  logical cores: {platform.os.cpu_count()}")
        print("  psutil: not installed")


def main() -> int:
    print(f"Python version: {sys.version}")
    print_cpu_information()

    try:
        import torch

        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA device: {torch.cuda.get_device_name(0)}")

        mps_available = bool(
            hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        )
        print(f"MPS available: {mps_available}")
    except ImportError:
        print("PyTorch version: not installed")
        print("CUDA available: unknown")
        print("MPS available: unknown")

    print(f"Transformers version: {package_version('transformers')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

