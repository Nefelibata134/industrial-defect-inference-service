from __future__ import annotations

import argparse
from pathlib import Path

import torch

from industrial_defect.config import load_project_config
from industrial_defect.training_data import SeverstalSegmentationDataset, build_dataloader


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect one segmentation training batch.")
    parser.add_argument("--config", default="configs/project.yaml")
    parser.add_argument("--data-root", default="data/raw/severstal")
    parser.add_argument("--manifest", default="data/manifests/severstal_v1.csv")
    parser.add_argument("--split", default="train", choices=("train", "val", "test"))
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument(
        "--image-size",
        type=int,
        nargs=2,
        metavar=("WIDTH", "HEIGHT"),
        help="Override the configured training image size.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_project_config(args.config)
    image_size = tuple(args.image_size) if args.image_size else config.training.image_size
    dataset = SeverstalSegmentationDataset(
        dataset_root=Path(args.data_root),
        manifest_path=Path(args.manifest),
        split=args.split,
        class_count=len(config.dataset.classes),
        image_size=(image_size[0], image_size[1]),
    )
    loader = build_dataloader(
        dataset,
        batch_size=args.batch_size,
        shuffle=args.split == "train",
        num_workers=args.workers,
        pin_memory=torch.cuda.is_available(),
        seed=config.seed,
    )
    batch = next(iter(loader))

    print("split:", args.split)
    print("dataset samples:", len(dataset))
    print("batch images:", tuple(batch["image"].shape), batch["image"].dtype)
    print("batch masks:", tuple(batch["mask"].shape), batch["mask"].dtype)
    print("mask values:", torch.unique(batch["mask"]).tolist())
    print("image ids:", list(batch["image_id"]))


if __name__ == "__main__":
    main()
