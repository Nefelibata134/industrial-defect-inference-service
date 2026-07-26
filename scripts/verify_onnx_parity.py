from __future__ import annotations

import argparse
import json
from pathlib import Path

import onnxruntime as ort
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from industrial_defect.config import load_project_config
from industrial_defect.metrics import SegmentationMetrics
from industrial_defect.onnx_export import (
    load_inference_model,
    sha256_file,
)
from industrial_defect.training_data import SeverstalSegmentationDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify PyTorch and ONNX Runtime output parity."
    )
    parser.add_argument("--config", default="configs/project.yaml")
    parser.add_argument("--checkpoint")
    parser.add_argument("--model")
    parser.add_argument("--data-root", default="data/raw/severstal")
    parser.add_argument("--manifest", default="data/manifests/severstal_v1.csv")
    parser.add_argument("--split", default="val")
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-abs-tolerance", type=float, default=1e-3)
    parser.add_argument("--mean-abs-tolerance", type=float, default=1e-5)
    parser.add_argument("--mask-mismatch-tolerance", type=float, default=1e-6)
    parser.add_argument("--metric-delta-tolerance", type=float, default=1e-6)
    parser.add_argument("--report", default="outputs/reports/onnx_parity.json")
    return parser.parse_args()


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(f"{path.suffix}.tmp")
    temporary_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    temporary_path.replace(path)


def main() -> None:
    args = parse_args()
    if args.samples <= 0 or args.batch_size <= 0:
        raise ValueError("samples and batch size must be positive")

    config = load_project_config(args.config)
    checkpoint_path = Path(args.checkpoint or config.deployment.checkpoint)
    model_path = Path(args.model or config.deployment.onnx_model)
    model, checkpoint = load_inference_model(config, checkpoint_path)
    width, height = config.training.image_size
    class_count = len(config.dataset.classes)

    dataset = SeverstalSegmentationDataset(
        dataset_root=args.data_root,
        manifest_path=args.manifest,
        split=args.split,
        class_count=class_count,
        image_size=(width, height),
    )
    sample_count = min(args.samples, len(dataset))
    loader = DataLoader(
        Subset(dataset, range(sample_count)),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    session = ort.InferenceSession(
        str(model_path),
        providers=["CPUExecutionProvider"],
    )
    session_inputs = session.get_inputs()
    session_outputs = session.get_outputs()
    if len(session_inputs) != 1 or len(session_outputs) != 1:
        raise ValueError("expected exactly one ONNX input and one ONNX output")
    input_name = session_inputs[0].name
    output_name = session_outputs[0].name
    if input_name != config.deployment.input_name:
        raise ValueError(f"unexpected ONNX input name: {input_name}")
    if output_name != config.deployment.output_name:
        raise ValueError(f"unexpected ONNX output name: {output_name}")

    threshold = config.deployment.threshold
    pytorch_metrics = SegmentationMetrics(class_count, threshold=threshold)
    onnx_metrics = SegmentationMetrics(class_count, threshold=threshold)
    absolute_error_sum = 0.0
    compared_values = 0
    max_abs_error = 0.0
    mismatched_masks = 0
    compared_masks = 0
    image_ids: list[str] = []

    with torch.inference_mode():
        for batch in tqdm(loader, desc=f"ONNX parity ({args.split})"):
            images = batch["image"].to(dtype=torch.float32)
            targets = batch["mask"]
            pytorch_logits = model(images).cpu()
            onnx_output = session.run(
                [output_name],
                {input_name: images.numpy()},
            )[0]
            onnx_logits = torch.from_numpy(onnx_output)
            if pytorch_logits.shape != onnx_logits.shape:
                raise ValueError(
                    "PyTorch and ONNX output shapes differ: "
                    f"{tuple(pytorch_logits.shape)} != {tuple(onnx_logits.shape)}"
                )

            absolute_error = torch.abs(pytorch_logits - onnx_logits)
            absolute_error_sum += float(absolute_error.sum().item())
            compared_values += absolute_error.numel()
            max_abs_error = max(max_abs_error, float(absolute_error.max().item()))

            pytorch_masks = torch.sigmoid(pytorch_logits) >= threshold
            onnx_masks = torch.sigmoid(onnx_logits) >= threshold
            mismatched_masks += int((pytorch_masks != onnx_masks).sum().item())
            compared_masks += pytorch_masks.numel()

            pytorch_metrics.update(pytorch_logits, targets)
            onnx_metrics.update(onnx_logits, targets)
            image_ids.extend(batch["image_id"])

    mean_abs_error = absolute_error_sum / compared_values
    mask_mismatch_ratio = mismatched_masks / compared_masks
    pytorch_quality = pytorch_metrics.compute()
    onnx_quality = onnx_metrics.compute()
    macro_dice_delta = abs(
        float(pytorch_quality["macro_dice"]) - float(onnx_quality["macro_dice"])
    )
    passed = (
        max_abs_error <= args.max_abs_tolerance
        and mean_abs_error <= args.mean_abs_tolerance
        and mask_mismatch_ratio <= args.mask_mismatch_tolerance
        and macro_dice_delta <= args.metric_delta_tolerance
    )

    report = {
        "status": "pass" if passed else "fail",
        "artifacts": {
            "checkpoint": str(checkpoint_path),
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "checkpoint_epoch": int(checkpoint["epoch"]),
            "onnx_model": str(model_path),
            "onnx_sha256": sha256_file(model_path),
        },
        "contract": {
            "input": {
                "name": input_name,
                "dtype": "float32",
                "shape": ["batch", 3, height, width],
            },
            "output": {
                "name": output_name,
                "dtype": "float32",
                "shape": ["batch", class_count, height, width],
            },
            "postprocess": "sigmoid",
            "threshold": threshold,
        },
        "regression_set": {
            "split": args.split,
            "samples": sample_count,
            "batch_size": args.batch_size,
            "image_ids": image_ids,
        },
        "runtime": {
            "torch": torch.__version__,
            "onnxruntime": ort.__version__,
            "providers": session.get_providers(),
        },
        "numerical_parity": {
            "max_abs_error": max_abs_error,
            "mean_abs_error": mean_abs_error,
            "mask_mismatch_pixels": mismatched_masks,
            "mask_mismatch_ratio": mask_mismatch_ratio,
        },
        "quality_parity": {
            "pytorch": pytorch_quality,
            "onnxruntime": onnx_quality,
            "macro_dice_abs_delta": macro_dice_delta,
        },
        "criteria": {
            "max_abs_tolerance": args.max_abs_tolerance,
            "mean_abs_tolerance": args.mean_abs_tolerance,
            "mask_mismatch_tolerance": args.mask_mismatch_tolerance,
            "metric_delta_tolerance": args.metric_delta_tolerance,
        },
    }
    report_path = Path(args.report)
    write_json(report_path, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    print("report:", report_path)
    if not passed:
        raise SystemExit("ONNX parity criteria failed")


if __name__ == "__main__":
    main()
