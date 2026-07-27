from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from industrial_defect.artifacts import sha256_file
from industrial_defect.benchmarking import (
    cuda_memory_info,
    current_process_gpu_memory_mib,
    latency_summary,
    system_metadata,
)
from industrial_defect.config import load_project_config
from industrial_defect.inference_data import load_inference_batch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark PyTorch CUDA or ONNX Runtime CUDA with CPU tensor I/O."
    )
    parser.add_argument(
        "--backend",
        choices=("pytorch-cuda", "onnxruntime-cuda"),
        required=True,
    )
    parser.add_argument("--config", default="configs/project.yaml")
    parser.add_argument("--data-root", default="data/raw/severstal")
    parser.add_argument("--manifest", default="data/manifests/severstal_v1.csv")
    parser.add_argument("--split", default="val")
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=None)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--runs", type=int, default=500)
    parser.add_argument("--report", required=True)
    return parser.parse_args()


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    temporary.replace(path)


def _benchmark_pytorch(
    *,
    config: Any,
    inputs: np.ndarray,
    warmup: int,
    runs: int,
) -> tuple[np.ndarray, list[float], dict[str, Any]]:
    import torch

    from industrial_defect.onnx_export import load_inference_model

    if not torch.cuda.is_available():
        raise RuntimeError("PyTorch CUDA is not available")

    gc.collect()
    torch.cuda.empty_cache()
    free_before, total_memory = cuda_memory_info()
    model, _ = load_inference_model(config, config.deployment.checkpoint)
    model = model.to("cuda")
    torch.cuda.reset_peak_memory_stats()

    def infer() -> np.ndarray:
        tensor = torch.from_numpy(inputs).to("cuda")
        with torch.inference_mode():
            output = model(tensor)
        result = output.cpu().numpy()
        torch.cuda.synchronize()
        return result

    for _ in range(warmup):
        infer()
    free_after_warmup, _ = cuda_memory_info()

    latencies: list[float] = []
    output = np.empty(0, dtype=np.float32)
    for _ in range(runs):
        started = time.perf_counter()
        output = infer()
        latencies.append((time.perf_counter() - started) * 1000.0)

    runtime = {
        "framework_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "gpu_process_memory_mib": current_process_gpu_memory_mib(),
        "gpu_memory_delta_bytes": max(0, free_before - free_after_warmup),
        "gpu_memory_total_bytes": total_memory,
        "gpu_memory_measurement": "cudaMemGetInfo delta before load to after warmup",
        "torch_peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "torch_peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
    }
    return output, latencies, runtime


def _benchmark_onnxruntime(
    *,
    config: Any,
    inputs: np.ndarray,
    warmup: int,
    runs: int,
) -> tuple[np.ndarray, list[float], dict[str, Any]]:
    import onnxruntime as ort

    ort.preload_dlls()
    gc.collect()
    free_before, total_memory = cuda_memory_info()
    session = ort.InferenceSession(
        config.deployment.onnx_model,
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )
    providers = session.get_providers()
    if not providers or providers[0] != "CUDAExecutionProvider":
        raise RuntimeError(f"ONNX Runtime CUDA provider is not active: {providers}")

    def infer() -> np.ndarray:
        return session.run(
            [config.deployment.output_name],
            {config.deployment.input_name: inputs},
        )[0]

    for _ in range(warmup):
        infer()
    free_after_warmup, _ = cuda_memory_info()

    latencies: list[float] = []
    output = np.empty(0, dtype=np.float32)
    for _ in range(runs):
        started = time.perf_counter()
        output = infer()
        latencies.append((time.perf_counter() - started) * 1000.0)

    runtime = {
        "framework_version": ort.__version__,
        "providers": providers,
        "gpu_process_memory_mib": current_process_gpu_memory_mib(),
        "gpu_memory_delta_bytes": max(0, free_before - free_after_warmup),
        "gpu_memory_total_bytes": total_memory,
        "gpu_memory_measurement": "cudaMemGetInfo delta before load to after warmup",
    }
    return output, latencies, runtime


def main() -> None:
    args = parse_args()
    if args.warmup < 0 or args.runs <= 0:
        raise ValueError("warmup must be non-negative and runs must be positive")

    config = load_project_config(args.config)
    batch_sizes = (
        tuple(args.batch_sizes)
        if args.batch_sizes is not None
        else tuple(sorted(set(config.deployment.tensorrt_profile_batch)))
    )
    if any(batch_size <= 0 for batch_size in batch_sizes):
        raise ValueError("batch sizes must be positive")

    results: list[dict[str, Any]] = []
    runtime: dict[str, Any] = {}
    for batch_size in batch_sizes:
        inputs, image_ids = load_inference_batch(
            args.data_root,
            args.manifest,
            split=args.split,
            batch_size=batch_size,
            image_size=config.training.image_size,
        )
        print(f"benchmarking {args.backend} batch={batch_size}...")
        if args.backend == "pytorch-cuda":
            output, latencies, runtime = _benchmark_pytorch(
                config=config,
                inputs=inputs,
                warmup=args.warmup,
                runs=args.runs,
            )
            artifact_path = Path(config.deployment.checkpoint)
        else:
            output, latencies, runtime = _benchmark_onnxruntime(
                config=config,
                inputs=inputs,
                warmup=args.warmup,
                runs=args.runs,
            )
            artifact_path = Path(config.deployment.onnx_model)

        expected_shape = (
            batch_size,
            len(config.dataset.classes),
            config.training.image_size[1],
            config.training.image_size[0],
        )
        if output.shape != expected_shape or output.dtype != np.float32:
            raise RuntimeError(
                f"unexpected output contract: shape={output.shape}, dtype={output.dtype}"
            )

        results.append(
            {
                "batch_size": batch_size,
                "image_ids": image_ids,
                "input_shape": list(inputs.shape),
                "output_shape": list(output.shape),
                "latency": latency_summary(latencies, batch_size=batch_size),
                "runtime": runtime,
            }
        )

    report = {
        "status": "pass",
        "backend": args.backend,
        "timing_boundary": "CPU float32 tensor input to CPU float32 logits output",
        "preprocessing_included": False,
        "postprocessing_included": False,
        "artifact_path": str(artifact_path),
        "artifact_sha256": sha256_file(artifact_path),
        "warmup_runs": args.warmup,
        "test_runs": args.runs,
        "system": system_metadata(),
        "results": results,
    }
    report_path = Path(args.report)
    write_json(report_path, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    print("report:", report_path)


if __name__ == "__main__":
    main()
