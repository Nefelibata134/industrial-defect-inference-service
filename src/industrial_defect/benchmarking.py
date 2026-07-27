from __future__ import annotations

import ctypes
import ctypes.util
import os
import platform
import site
import subprocess
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np


def latency_summary(
    values_ms: Sequence[float],
    *,
    batch_size: int,
) -> dict[str, float]:
    if not values_ms:
        raise ValueError("latency values must be non-empty")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    values = np.asarray(values_ms, dtype=np.float64)
    mean_ms = float(values.mean())
    return {
        "mean_ms": mean_ms,
        "min_ms": float(values.min()),
        "max_ms": float(values.max()),
        "p50_ms": float(np.percentile(values, 50)),
        "p95_ms": float(np.percentile(values, 95)),
        "p99_ms": float(np.percentile(values, 99)),
        "throughput_images_per_second": batch_size * 1000.0 / mean_ms,
    }


def gpu_metadata() -> dict[str, Any]:
    command = [
        "nvidia-smi",
        "--query-gpu=name,driver_version,memory.total",
        "--format=csv,noheader,nounits",
    ]
    result = subprocess.run(
        command,
        capture_output=True,
        check=False,
        text=True,
    )
    if result.returncode != 0 or not result.stdout.strip():
        return {"available": False}

    name, driver_version, memory_mib = (
        item.strip() for item in result.stdout.splitlines()[0].split(",", maxsplit=2)
    )
    return {
        "available": True,
        "name": name,
        "driver_version": driver_version,
        "memory_total_mib": int(memory_mib),
    }


def current_process_gpu_memory_mib() -> int | None:
    command = [
        "nvidia-smi",
        "--query-compute-apps=pid,used_gpu_memory",
        "--format=csv,noheader,nounits",
    ]
    result = subprocess.run(
        command,
        capture_output=True,
        check=False,
        text=True,
    )
    if result.returncode != 0:
        return None

    current_pid = os.getpid()
    for line in result.stdout.splitlines():
        fields = [item.strip() for item in line.split(",", maxsplit=1)]
        if len(fields) != 2 or not fields[0].isdigit():
            continue
        if int(fields[0]) != current_pid or not fields[1].isdigit():
            continue
        return int(fields[1])
    return None


def cuda_memory_info() -> tuple[int, int]:
    candidates: list[str] = []
    discovered = ctypes.util.find_library("cudart")
    if discovered:
        candidates.append(discovered)
    for package_root in site.getsitepackages():
        candidates.extend(
            str(path)
            for path in Path(package_root).glob(
                "nvidia/cuda_runtime/lib/libcudart.so*"
            )
        )
    candidates.extend(("libcudart.so.13", "libcudart.so.12"))

    library: ctypes.CDLL | None = None
    for candidate in dict.fromkeys(candidates):
        try:
            library = ctypes.CDLL(candidate)
            break
        except OSError:
            continue
    if library is None:
        raise RuntimeError("CUDA Runtime library could not be loaded")

    free_bytes = ctypes.c_size_t()
    total_bytes = ctypes.c_size_t()
    status = library.cudaMemGetInfo(
        ctypes.byref(free_bytes),
        ctypes.byref(total_bytes),
    )
    if status != 0:
        raise RuntimeError(f"cudaMemGetInfo failed with CUDA error {status}")
    return free_bytes.value, total_bytes.value


def system_metadata() -> dict[str, Any]:
    return {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "gpu": gpu_metadata(),
    }
