from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import torch

from industrial_defect.config import load_project_config
from industrial_defect.model import build_segmentation_model
from industrial_defect.training_data import SeverstalSegmentationDataset
from industrial_defect.visualization import (
    build_comparison_panel,
    denormalize_image,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render ground-truth, prediction, and pixel-error comparison panels."
    )
    parser.add_argument(
        "--analysis-report",
        default="outputs/reports/error_analysis.json",
    )
    parser.add_argument("--config", default="configs/project.yaml")
    parser.add_argument("--data-root", default="data/raw/severstal")
    parser.add_argument("--manifest", default="data/manifests/severstal_v1.csv")
    parser.add_argument(
        "--output-dir",
        default="outputs/error_analysis",
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--alpha", type=float, default=0.75)
    return parser.parse_args()


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return torch.device(requested)


def load_report(path: Path) -> dict[str, Any]:
    report = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "checkpoint",
        "split",
        "image_size",
        "threshold",
        "focus_class",
        "selected_examples",
    }
    missing = sorted(required - report.keys())
    if missing:
        raise ValueError(f"analysis report is missing fields: {missing}")
    return report


def collect_examples(
    selected_examples: dict[str, list[dict[str, Any]]],
) -> tuple[dict[str, list[tuple[str, dict[str, Any]]]], set[str]]:
    grouped: dict[str, list[tuple[str, dict[str, Any]]]] = defaultdict(list)
    image_ids: set[str] = set()
    for group_name, examples in selected_examples.items():
        for example in examples:
            image_id = str(example["image_id"])
            grouped[image_id].append((group_name, example))
            image_ids.add(image_id)
    return dict(grouped), image_ids


def format_summary(example: dict[str, Any]) -> str:
    dice = example.get("dice")
    dice_text = "n/a" if dice is None else f"{float(dice):.3f}"
    return (
        f"Dice={dice_text} | target={example['target_pixels']} px | "
        f"prediction={example['predicted_pixels']} px"
    )


def main() -> None:
    args = parse_args()
    if not 0.0 <= args.alpha <= 1.0:
        raise ValueError("alpha must be between 0 and 1")

    report = load_report(Path(args.analysis_report))
    config = load_project_config(args.config)
    class_count = len(config.dataset.classes)
    focus_class = int(report["focus_class"])
    if not 1 <= focus_class <= class_count:
        raise ValueError(f"focus class must be between 1 and {class_count}")

    image_size = tuple(int(value) for value in report["image_size"])
    if len(image_size) != 2:
        raise ValueError("analysis report image_size must contain width and height")

    grouped_examples, selected_image_ids = collect_examples(report["selected_examples"])
    if not selected_image_ids:
        raise ValueError("analysis report contains no selected examples")

    dataset = SeverstalSegmentationDataset(
        dataset_root=args.data_root,
        manifest_path=args.manifest,
        split=str(report["split"]),
        class_count=class_count,
        image_size=(image_size[0], image_size[1]),
    )
    index_by_image_id = {
        image_id: index for index, image_id in enumerate(dataset.image_ids)
    }
    missing = sorted(selected_image_ids - index_by_image_id.keys())
    if missing:
        raise ValueError(f"selected images are missing from the report split: {missing}")

    device = resolve_device(args.device)
    checkpoint_path = Path(str(report["checkpoint"]))
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = build_segmentation_model(
        config.training.model,
        class_count=class_count,
        pretrained=False,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    output_dir = Path(args.output_dir) / f"defect_{focus_class}"
    output_dir.mkdir(parents=True, exist_ok=True)
    threshold = float(report["threshold"])
    class_index = focus_class - 1
    class_name = config.dataset.classes[class_index]
    amp_enabled = config.training.amp and device.type == "cuda"

    with torch.inference_mode():
        for image_id in sorted(selected_image_ids):
            sample = dataset[index_by_image_id[image_id]]
            image_tensor = sample["image"]
            with torch.autocast(device_type=device.type, enabled=amp_enabled):
                logits = model(image_tensor.unsqueeze(0).to(device))

            probabilities = torch.sigmoid(logits)
            predicted_mask = (
                probabilities[0, class_index].float() >= threshold
            ).cpu().numpy()
            target_mask = (sample["mask"][class_index] >= 0.5).numpy()
            image = denormalize_image(image_tensor)

            for group_name, example in grouped_examples[image_id]:
                panel = build_comparison_panel(
                    image,
                    target_mask,
                    predicted_mask,
                    class_name=class_name,
                    summary=format_summary(example),
                    alpha=args.alpha,
                )
                group_dir = output_dir / group_name
                group_dir.mkdir(parents=True, exist_ok=True)
                output_path = group_dir / f"{Path(image_id).stem}_comparison.jpg"
                if not cv2.imwrite(
                    str(output_path),
                    cv2.cvtColor(panel, cv2.COLOR_RGB2BGR),
                ):
                    raise RuntimeError(f"failed to save image: {output_path}")
                print(f"{group_name:>15}  {image_id}  ->  {output_path}")

    print("output:", output_dir)


if __name__ == "__main__":
    main()
