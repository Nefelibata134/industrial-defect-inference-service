from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort

from industrial_defect.artifacts import sha256_file
from industrial_defect.benchmarking import (
    current_process_gpu_memory_mib,
    latency_summary,
    system_metadata,
)
from industrial_defect.config import load_project_config
from industrial_defect.deployment_validation import validate_backend_parity
from industrial_defect.inference_data import (
    load_inference_batch,
    probability_to_logit_threshold,
)
from industrial_defect.tensorrt_runtime import TensorRTRunner


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate and benchmark FP32/FP16 TensorRT engines."
    )
    parser.add_argument("--config", default="configs/project.yaml")
    parser.add_argument("--data-root", default="data/raw/severstal")
    parser.add_argument("--manifest", default="data/manifests/severstal_v1.csv")
    parser.add_argument("--split", default="val")
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--runs", type=int, default=500)
    parser.add_argument(
        "--report",
        default="outputs/reports/tensorrt_benchmark.json",
    )
    return parser.parse_args()


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    if args.warmup < 0 or args.runs <= 0:
        raise ValueError("warmup must be non-negative and runs must be positive")

    config = load_project_config(args.config)
    probability_threshold = config.deployment.threshold
    logit_threshold = probability_to_logit_threshold(probability_threshold)
    engines = {
        "fp32": Path(config.deployment.tensorrt_fp32_engine),
        "fp16": Path(config.deployment.tensorrt_fp16_engine),
    }
    session = ort.InferenceSession(
        config.deployment.onnx_model,
        providers=["CPUExecutionProvider"],
    )
    batch_sizes = tuple(sorted(set(config.deployment.tensorrt_profile_batch)))
    results: list[dict[str, object]] = []
    tensorrt_version = ""

    for batch_size in batch_sizes:
        inputs, image_ids = load_inference_batch(
            args.data_root,
            args.manifest,
            split=args.split,
            batch_size=batch_size,
            image_size=config.training.image_size,
        )
        reference = session.run(
            [config.deployment.output_name],
            {config.deployment.input_name: inputs},
        )[0]
        reference_mask = reference >= logit_threshold

        for precision, engine_path in engines.items():
            print(f"benchmarking {precision} batch={batch_size}...")
            with TensorRTRunner(engine_path) as runner:
                tensorrt_version = runner.trt.__version__
                output, _ = runner.infer(inputs)
                difference = np.abs(reference - output)
                mismatch = int(
                    np.count_nonzero(
                        reference_mask
                        != (output >= logit_threshold)
                    )
                )
                mismatch_rate = mismatch / reference_mask.size
                for _ in range(args.warmup):
                    runner.infer(inputs)

                gpu_latencies: list[float] = []
                wall_latencies: list[float] = []
                for _ in range(args.runs):
                    started = time.perf_counter()
                    _, gpu_ms = runner.infer(inputs)
                    wall_latencies.append(
                        (time.perf_counter() - started) * 1000.0
                    )
                    gpu_latencies.append(gpu_ms)

                engine_device_memory_bytes = int(
                    runner.engine.device_memory_size_v2
                )
                io_buffer_bytes = int(inputs.nbytes + output.nbytes)
                gpu_process_memory_mib = current_process_gpu_memory_mib()

            results.append(
                {
                    "precision": precision,
                    "batch_size": batch_size,
                    "image_ids": image_ids,
                    "engine_path": str(engine_path),
                    "engine_sha256": sha256_file(engine_path),
                    "engine_size_bytes": engine_path.stat().st_size,
                    "engine_device_memory_bytes": engine_device_memory_bytes,
                    "io_buffer_bytes": io_buffer_bytes,
                    "context_and_io_bytes": (
                        engine_device_memory_bytes + io_buffer_bytes
                    ),
                    "gpu_process_memory_mib": gpu_process_memory_mib,
                    "memory_capacity_note": (
                        "context_and_io_bytes excludes engine weights, CUDA "
                        "context overhead, and allocator caching"
                    ),
                    "max_abs_error_vs_ort": float(difference.max()),
                    "mean_abs_error_vs_ort": float(difference.mean()),
                    "mask_mismatch_pixels": mismatch,
                    "mask_mismatch_rate": mismatch_rate,
                    "acceptance": validate_backend_parity(
                        precision,
                        max_abs_error=float(difference.max()),
                        mean_abs_error=float(difference.mean()),
                        mask_mismatch_rate=mismatch_rate,
                    ),
                    "gpu_inference": latency_summary(
                        gpu_latencies,
                        batch_size=batch_size,
                    ),
                    "wall_inference": latency_summary(
                        wall_latencies,
                        batch_size=batch_size,
                    ),
                }
            )

    passed = all(
        result["acceptance"]["passed"]
        for result in results
    )
    report = {
        "status": "pass" if passed else "fail",
        "tensorrt_version": tensorrt_version,
        "onnx_path": config.deployment.onnx_model,
        "onnx_sha256": sha256_file(config.deployment.onnx_model),
        "probability_threshold": probability_threshold,
        "logit_threshold": logit_threshold,
        "warmup_runs": args.warmup,
        "test_runs": args.runs,
        "system": system_metadata(),
        "results": results,
    }
    report_path = Path(args.report)
    write_json(report_path, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    print("report:", report_path)
    if not passed:
        raise SystemExit("TensorRT benchmark failed acceptance criteria")


if __name__ == "__main__":
    main()
