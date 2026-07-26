from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from industrial_defect.config import load_project_config
from industrial_defect.onnx_export import (
    export_onnx_model,
    load_inference_model,
    sha256_file,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export the promoted segmentation model.")
    parser.add_argument("--config", default="configs/project.yaml")
    parser.add_argument("--checkpoint")
    parser.add_argument("--output")
    parser.add_argument("--batch-size", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_project_config(args.config)
    checkpoint_path = Path(args.checkpoint or config.deployment.checkpoint)
    output_path = Path(args.output or config.deployment.onnx_model)
    if args.batch_size <= 0:
        raise ValueError("batch size must be positive")

    model, checkpoint = load_inference_model(config, checkpoint_path)
    width, height = config.training.image_size
    example_input = torch.zeros(
        (args.batch_size, 3, height, width),
        dtype=torch.float32,
    )
    metadata = {
        "architecture": config.training.model,
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "classes": ",".join(config.dataset.classes),
        "checkpoint_epoch": str(checkpoint["epoch"]),
        "input_height": str(height),
        "input_width": str(width),
        "postprocess": "sigmoid",
        "threshold": f"{config.deployment.threshold:.2f}",
    }
    summary = export_onnx_model(
        model,
        example_input,
        output_path,
        input_name=config.deployment.input_name,
        output_name=config.deployment.output_name,
        opset=config.deployment.onnx_opset,
        dynamic_batch=config.deployment.dynamic_batch,
        metadata=metadata,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
