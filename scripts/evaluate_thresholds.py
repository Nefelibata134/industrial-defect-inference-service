from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
from tqdm import tqdm

from industrial_defect.config import load_project_config
from industrial_defect.model import build_segmentation_model
from industrial_defect.thresholds import ThresholdSweepMetrics, select_best_thresholds
from industrial_defect.training_data import SeverstalSegmentationDataset, build_dataloader


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sweep segmentation thresholds on a labeled split."
    )
    parser.add_argument("--config", default="configs/project.yaml")
    parser.add_argument("--data-root", default="data/raw/severstal")
    parser.add_argument("--manifest", default="data/manifests/severstal_v1.csv")
    parser.add_argument("--split", default="val")
    parser.add_argument("--checkpoint", default="models/best_unet_resnet18.pt")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--image-size", type=int, nargs=2, metavar=("WIDTH", "HEIGHT"))
    parser.add_argument(
        "--thresholds",
        type=float,
        nargs="+",
        default=(0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7),
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--output", default="outputs/reports/threshold_sweep.json")
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


def main() -> None:
    args = parse_args()
    config = load_project_config(args.config)
    device = resolve_device(args.device)
    class_count = len(config.dataset.classes)
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

    meter = ThresholdSweepMetrics(class_count, args.thresholds)
    amp_enabled = config.training.amp and device.type == "cuda"
    with torch.inference_mode():
        for batch in tqdm(loader, desc=f"threshold sweep ({args.split})"):
            images = batch["image"].to(device, non_blocking=True)
            masks = batch["mask"].to(device, non_blocking=True)
            with torch.autocast(
                device_type=device.type,
                enabled=amp_enabled,
            ):
                logits = model(images)
            meter.update(logits.float(), masks)

    results = meter.compute()
    best_macro = max(
        results,
        key=lambda result: (
            float(result["macro_dice"]),
            float(result["threshold"]),
        ),
    )
    best_per_class = select_best_thresholds(results)
    payload = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": file_sha256(checkpoint_path),
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "split": args.split,
        "samples": len(dataset),
        "image_size": list(image_size),
        "thresholds": list(meter.thresholds),
        "results": results,
        "best_macro": {
            "threshold": best_macro["threshold"],
            "dice": best_macro["macro_dice"],
        },
        "best_per_class": best_per_class,
    }
    output_path = Path(args.output)
    write_json(output_path, payload)

    header = "threshold  macro  " + "  ".join(
        f"class_{index}" for index in range(1, class_count + 1)
    )
    print(header)
    for result in results:
        scores = result["per_class_dice"]
        if not isinstance(scores, list):
            raise TypeError("per-class Dice scores must be a list")
        score_text = "  ".join(f"{float(score):.4f}" for score in scores)
        print(
            f"{float(result['threshold']):>9.2f}  "
            f"{float(result['macro_dice']):.4f}  "
            f"{score_text}"
        )

    print("\nbest macro:", payload["best_macro"])
    print("best per class:")
    for result in best_per_class:
        print(
            f"- class_{result['class_id']}: "
            f"threshold={float(result['threshold']):.2f} "
            f"dice={float(result['dice']):.4f}"
        )
    print("saved:", output_path)


if __name__ == "__main__":
    main()
