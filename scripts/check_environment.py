from __future__ import annotations

import importlib
import platform
import sys
from importlib.metadata import PackageNotFoundError, version


def package_version(package: str) -> str:
    try:
        return version(package)
    except PackageNotFoundError:
        return "not installed"


def main() -> None:
    print("Industrial Defect Inference Service - Environment")
    print(f"python: {platform.python_version()}")
    print(f"executable: {sys.executable}")
    print(f"platform: {platform.platform()}")
    print(f"architecture: {platform.machine()}")

    for package in (
        "numpy",
        "opencv-python-headless",
        "PyYAML",
        "albumentations",
        "segmentation-models-pytorch",
        "torch",
        "onnx",
        "onnxruntime-gpu",
        "tensorrt",
        "tritonclient",
        "fastapi",
    ):
        print(f"{package}: {package_version(package)}")

    try:
        torch = importlib.import_module("torch")
    except ImportError:
        print("torch cuda available: unavailable (torch is not installed)")
    else:
        print(f"torch cuda available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"cuda device: {torch.cuda.get_device_name(0)}")


if __name__ == "__main__":
    main()
