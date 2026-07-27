from __future__ import annotations

import argparse
import json
from pathlib import Path

from industrial_defect.config import load_project_config
from industrial_defect.tensorrt_engine import build_tensorrt_engine


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build reproducible FP32 and FP16 TensorRT engines."
    )
    parser.add_argument("--config", default="configs/project.yaml")
    parser.add_argument(
        "--precision",
        choices=("fp32", "fp16", "both"),
        default="both",
    )
    parser.add_argument(
        "--report",
        default="outputs/reports/tensorrt_build.json",
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
    config = load_project_config(args.config)
    width, height = config.training.image_size
    profile_batch = config.deployment.tensorrt_profile_batch
    engine_paths = {
        "fp32": config.deployment.tensorrt_fp32_engine,
        "fp16": config.deployment.tensorrt_fp16_engine,
    }
    precisions = ("fp32", "fp16") if args.precision == "both" else (args.precision,)

    builds = []
    for precision in precisions:
        print(f"building TensorRT {precision.upper()} engine...")
        result = build_tensorrt_engine(
            config.deployment.onnx_model,
            engine_paths[precision],
            precision=precision,
            input_name=config.deployment.input_name,
            profile_batch=profile_batch,
            channels=3,
            height=height,
            width=width,
            workspace_gib=config.deployment.tensorrt_workspace_gib,
        )
        builds.append(result)
        print(
            f"saved {result['engine_path']} "
            f"({result['engine_size_bytes'] / 1024**2:.2f} MiB)"
        )

    report = {
        "status": "pass",
        "builds": builds,
    }
    report_path = Path(args.report)
    write_json(report_path, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    print("report:", report_path)


if __name__ == "__main__":
    main()
