from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm

from industrial_defect.config import load_project_config
from industrial_defect.error_analysis import (
    aggregate_image_presence,
    compute_image_class_statistics,
)
from industrial_defect.metrics import SegmentationMetrics
from industrial_defect.model import build_segmentation_model
from industrial_defect.training_data import SeverstalSegmentationDataset, build_dataloader


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze pixel and image-level segmentation errors."
    )
    parser.add_argument("--config", default="configs/project.yaml")
    parser.add_argument("--data-root", default="data/raw/severstal")
    parser.add_argument("--manifest", default="data/manifests/severstal_v1.csv")
    parser.add_argument("--split", default="val")
    parser.add_argument("--checkpoint", default="models/best_unet_resnet18.pt")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--focus-class", type=int, default=2)
    parser.add_argument("--examples-per-group", type=int, default=5)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--image-size", type=int, nargs=2, metavar=("WIDTH", "HEIGHT"))
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--max-batches", type=int)
    parser.add_argument(
        "--output",
        default="outputs/reports/error_analysis.json",
    )
    parser.add_argument(
        "--per-image-output",
        default="outputs/reports/per_image_metrics.csv",
    )
    return parser.parse_args()


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return torch.device(requested)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(f"{path.suffix}.tmp")
    temporary_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    temporary_path.replace(path)


def optional_float(value: torch.Tensor) -> float | None:
    result = float(value.item())
    return None if math.isnan(result) else result


def presence_outcome(
    *,
    target_present: bool,
    prediction_present: bool,
) -> str:
    if target_present and prediction_present:
        return "true_positive"
    if not target_present and prediction_present:
        return "false_positive"
    if target_present and not prediction_present:
        return "false_negative"
    return "true_negative"


def select_examples(
    rows: list[dict[str, Any]],
    *,
    focus_class: int,
    limit: int,
) -> dict[str, list[dict[str, Any]]]:
    focus_rows = [row for row in rows if row["class_id"] == focus_class]
    target_positive = [row for row in focus_rows if row["target_present"]]
    overlap = [row for row in target_positive if row["image_has_overlap"]]
    false_negative = [
        row for row in focus_rows if row["presence_outcome"] == "false_negative"
    ]
    false_positive = [
        row for row in focus_rows if row["presence_outcome"] == "false_positive"
    ]
    mislocalized = [
        row
        for row in focus_rows
        if row["presence_outcome"] == "true_positive" and not row["image_has_overlap"]
    ]

    def dice_value(row: dict[str, Any]) -> float:
        value = row["dice"]
        return float(value) if value is not None else -1.0

    def compact(row: dict[str, Any]) -> dict[str, Any]:
        fields = (
            "image_id",
            "dice",
            "iou",
            "target_pixels",
            "predicted_pixels",
            "false_positive_pixels",
            "false_negative_pixels",
            "presence_outcome",
            "image_has_overlap",
        )
        return {field: row[field] for field in fields}

    return {
        "best_overlap": [
            compact(row)
            for row in sorted(overlap, key=dice_value, reverse=True)[:limit]
        ],
        "worst_positive": [
            compact(row) for row in sorted(target_positive, key=dice_value)[:limit]
        ],
        "false_negative": [
            compact(row)
            for row in sorted(
                false_negative,
                key=lambda row: int(row["target_pixels"]),
                reverse=True,
            )[:limit]
        ],
        "false_positive": [
            compact(row)
            for row in sorted(
                false_positive,
                key=lambda row: int(row["predicted_pixels"]),
                reverse=True,
            )[:limit]
        ],
        "mislocalized": [
            compact(row)
            for row in sorted(
                mislocalized,
                key=lambda row: int(row["target_pixels"]),
                reverse=True,
            )[:limit]
        ],
    }


def write_per_image_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0]) if rows else []
    temporary_path = path.with_suffix(f"{path.suffix}.tmp")
    with temporary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    temporary_path.replace(path)


def main() -> None:
    args = parse_args()
    if not 0.0 < args.threshold < 1.0:
        raise ValueError("threshold must be between 0 and 1")
    if args.examples_per_group <= 0:
        raise ValueError("examples-per-group must be positive")
    if args.max_batches is not None and args.max_batches <= 0:
        raise ValueError("max-batches must be positive")

    config = load_project_config(args.config)
    class_names = config.dataset.classes
    class_count = len(class_names)
    if not 1 <= args.focus_class <= class_count:
        raise ValueError(f"focus-class must be between 1 and {class_count}")

    device = resolve_device(args.device)
    image_size = tuple(args.image_size) if args.image_size else config.training.image_size
    batch_size = args.batch_size or config.training.batch
    workers = config.training.num_workers if args.workers is None else args.workers

    dataset = SeverstalSegmentationDataset(
        dataset_root=args.data_root,
        manifest_path=args.manifest,
        split=args.split,
        class_count=class_count,
        image_size=(image_size[0], image_size[1]),
    )
    loader = build_dataloader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=device.type == "cuda",
    )

    checkpoint_path = Path(args.checkpoint)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = build_segmentation_model(
        config.training.model,
        class_count=class_count,
        pretrained=False,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    aggregate_meter = SegmentationMetrics(class_count, threshold=args.threshold)
    presence_counts = {
        name: torch.zeros(class_count, dtype=torch.int64)
        for name in (
            "true_positive",
            "false_positive",
            "false_negative",
            "true_negative",
            "has_overlap",
        )
    }
    rows: list[dict[str, Any]] = []
    processed_samples = 0
    amp_enabled = config.training.amp and device.type == "cuda"

    with torch.inference_mode():
        for batch_index, batch in enumerate(
            tqdm(loader, desc=f"error analysis ({args.split})")
        ):
            if args.max_batches is not None and batch_index >= args.max_batches:
                break

            images = batch["image"].to(device, non_blocking=True)
            masks = batch["mask"].to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, enabled=amp_enabled):
                logits = model(images)

            aggregate_meter.update(logits.float(), masks)
            probabilities = torch.sigmoid(logits).float()
            predictions = probabilities >= args.threshold
            targets = masks >= 0.5
            statistics = compute_image_class_statistics(predictions, targets)
            batch_presence = aggregate_image_presence(statistics)

            for name, counts in batch_presence.items():
                presence_counts[name] += counts.detach().cpu()

            cpu_statistics = {
                name: values.detach().cpu() for name, values in statistics.items()
            }
            image_ids = list(batch["image_id"])
            processed_samples += len(image_ids)

            for sample_index, image_id in enumerate(image_ids):
                for class_index, class_name in enumerate(class_names):
                    target_present = bool(
                        cpu_statistics["target_present"][sample_index, class_index]
                    )
                    prediction_present = bool(
                        cpu_statistics["prediction_present"][sample_index, class_index]
                    )
                    rows.append(
                        {
                            "image_id": image_id,
                            "class_id": class_index + 1,
                            "class_name": class_name,
                            "intersection": int(
                                cpu_statistics["intersection"][sample_index, class_index]
                            ),
                            "predicted_pixels": int(
                                cpu_statistics["predicted_pixels"][
                                    sample_index, class_index
                                ]
                            ),
                            "target_pixels": int(
                                cpu_statistics["target_pixels"][sample_index, class_index]
                            ),
                            "false_positive_pixels": int(
                                cpu_statistics["false_positive_pixels"][
                                    sample_index, class_index
                                ]
                            ),
                            "false_negative_pixels": int(
                                cpu_statistics["false_negative_pixels"][
                                    sample_index, class_index
                                ]
                            ),
                            "union": int(
                                cpu_statistics["union"][sample_index, class_index]
                            ),
                            "dice": optional_float(
                                cpu_statistics["dice"][sample_index, class_index]
                            ),
                            "iou": optional_float(
                                cpu_statistics["iou"][sample_index, class_index]
                            ),
                            "precision": optional_float(
                                cpu_statistics["precision"][sample_index, class_index]
                            ),
                            "recall": optional_float(
                                cpu_statistics["recall"][sample_index, class_index]
                            ),
                            "target_present": target_present,
                            "prediction_present": prediction_present,
                            "presence_outcome": presence_outcome(
                                target_present=target_present,
                                prediction_present=prediction_present,
                            ),
                            "image_has_overlap": bool(
                                cpu_statistics["image_has_overlap"][
                                    sample_index, class_index
                                ]
                            ),
                        }
                    )

    aggregate_metrics = aggregate_meter.compute()
    per_class: list[dict[str, Any]] = []
    for class_index, class_name in enumerate(class_names):
        true_positive = int(presence_counts["true_positive"][class_index])
        false_positive = int(presence_counts["false_positive"][class_index])
        false_negative = int(presence_counts["false_negative"][class_index])
        true_negative = int(presence_counts["true_negative"][class_index])
        target_positive = true_positive + false_negative
        target_negative = false_positive + true_negative
        per_class.append(
            {
                "class_id": class_index + 1,
                "class_name": class_name,
                "dice": aggregate_metrics["per_class_dice"][class_index],
                "iou": aggregate_metrics["per_class_iou"][class_index],
                "precision": aggregate_metrics["per_class_precision"][class_index],
                "recall": aggregate_metrics["per_class_recall"][class_index],
                "target_positive_images": target_positive,
                "predicted_positive_images": true_positive + false_positive,
                "image_true_positive": true_positive,
                "image_false_positive": false_positive,
                "image_false_negative": false_negative,
                "image_true_negative": true_negative,
                "image_false_negative_rate": (
                    false_negative / target_positive if target_positive else None
                ),
                "image_false_positive_rate": (
                    false_positive / target_negative if target_negative else None
                ),
                "images_with_overlap": int(
                    presence_counts["has_overlap"][class_index]
                ),
            }
        )

    payload = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": file_sha256(checkpoint_path),
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "split": args.split,
        "samples": processed_samples,
        "image_size": list(image_size),
        "threshold": args.threshold,
        "aggregate_metrics": aggregate_metrics,
        "per_class": per_class,
        "focus_class": args.focus_class,
        "selected_examples": select_examples(
            rows,
            focus_class=args.focus_class,
            limit=args.examples_per_group,
        ),
    }

    output_path = Path(args.output)
    per_image_path = Path(args.per_image_output)
    write_json(output_path, payload)
    write_per_image_csv(per_image_path, rows)

    print("checkpoint epoch:", payload["checkpoint_epoch"])
    print("samples:", processed_samples)
    print("threshold:", args.threshold)
    print("macro Dice:", f"{float(aggregate_metrics['macro_dice']):.4f}")
    print("class  Dice    IoU     Precision  Recall  image_FN  image_FP")
    for result in per_class:
        print(
            f"{result['class_id']:>5}  "
            f"{result['dice']:.4f}  "
            f"{result['iou']:.4f}  "
            f"{result['precision']:.4f}     "
            f"{result['recall']:.4f}  "
            f"{result['image_false_negative']:>8}  "
            f"{result['image_false_positive']:>8}"
        )
    print("report:", output_path)
    print("per-image metrics:", per_image_path)


if __name__ == "__main__":
    main()
