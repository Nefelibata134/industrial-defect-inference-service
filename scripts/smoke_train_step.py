from __future__ import annotations

import argparse
from pathlib import Path

import torch

from industrial_defect.config import load_project_config
from industrial_defect.loss import BCEDiceLoss
from industrial_defect.model import build_segmentation_model
from industrial_defect.training_data import SeverstalSegmentationDataset, build_dataloader


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one segmentation optimization step.")
    parser.add_argument("--config", default="configs/project.yaml")
    parser.add_argument("--data-root", default="data/raw/severstal")
    parser.add_argument("--manifest", default="data/manifests/severstal_v1.csv")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument(
        "--image-size",
        type=int,
        nargs=2,
        metavar=("WIDTH", "HEIGHT"),
        help="Override the configured training image size.",
    )
    parser.add_argument(
        "--pretrained",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser.parse_args()


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return torch.device(requested)


def main() -> None:
    args = parse_args()
    config = load_project_config(args.config)
    device = resolve_device(args.device)
    image_size = tuple(args.image_size) if args.image_size else config.training.image_size
    class_count = len(config.dataset.classes)

    dataset = SeverstalSegmentationDataset(
        dataset_root=Path(args.data_root),
        manifest_path=Path(args.manifest),
        split="train",
        class_count=class_count,
        image_size=(image_size[0], image_size[1]),
    )
    loader = build_dataloader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        seed=config.seed,
    )
    model = build_segmentation_model(
        config.training.model,
        class_count=class_count,
        pretrained=args.pretrained,
    ).to(device)
    criterion = BCEDiceLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    batch = next(iter(loader))
    images = batch["image"].to(device, non_blocking=True)
    targets = batch["mask"].to(device, non_blocking=True)

    model.train()
    optimizer.zero_grad(set_to_none=True)
    logits = model(images)
    loss = criterion(logits, targets)
    loss.backward()
    optimizer.step()

    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    print("device:", device)
    print("pretrained encoder:", args.pretrained)
    print("parameters:", f"{parameter_count:,}")
    print("input:", tuple(images.shape), images.dtype)
    print("target:", tuple(targets.shape), targets.dtype)
    print("logits:", tuple(logits.shape), logits.dtype)
    print("loss:", float(loss.detach()))
    if device.type == "cuda":
        print("max CUDA memory MB:", round(torch.cuda.max_memory_allocated() / 1024**2, 2))


if __name__ == "__main__":
    main()
