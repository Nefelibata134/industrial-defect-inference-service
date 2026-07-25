from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from industrial_defect.config import load_project_config
from industrial_defect.loss import BCEDiceLoss
from industrial_defect.model import build_segmentation_model
from industrial_defect.trainer import run_epoch
from industrial_defect.training_data import SeverstalSegmentationDataset, build_dataloader


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the segmentation baseline.")
    parser.add_argument("--config", default="configs/project.yaml")
    parser.add_argument("--data-root", default="data/raw/severstal")
    parser.add_argument("--manifest", default="data/manifests/severstal_v1.csv")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--image-size", type=int, nargs=2, metavar=("WIDTH", "HEIGHT"))
    parser.add_argument("--max-train-batches", type=int)
    parser.add_argument("--max-val-batches", type=int)
    parser.add_argument("--pretrained", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--checkpoint", default="models/best_unet_resnet18.pt")
    parser.add_argument("--history", default="outputs/reports/training_history.json")
    return parser.parse_args()


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return torch.device(requested)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(f"{path.suffix}.tmp")
    temporary_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    temporary_path.replace(path)


def save_checkpoint(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(f"{path.suffix}.tmp")
    torch.save(payload, temporary_path)
    temporary_path.replace(path)


def main() -> None:
    args = parse_args()
    config = load_project_config(args.config)
    seed_everything(config.seed)
    device = resolve_device(args.device)
    image_size = tuple(args.image_size) if args.image_size else config.training.image_size
    batch_size = args.batch_size or config.training.batch
    workers = config.training.num_workers if args.workers is None else args.workers
    epochs = args.epochs or config.training.epochs
    class_count = len(config.dataset.classes)

    train_dataset = SeverstalSegmentationDataset(
        dataset_root=args.data_root,
        manifest_path=args.manifest,
        split="train",
        class_count=class_count,
        image_size=(image_size[0], image_size[1]),
    )
    val_dataset = SeverstalSegmentationDataset(
        dataset_root=args.data_root,
        manifest_path=args.manifest,
        split="val",
        class_count=class_count,
        image_size=(image_size[0], image_size[1]),
    )
    train_loader = build_dataloader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=device.type == "cuda",
        drop_last=True,
        seed=config.seed,
    )
    val_loader = build_dataloader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=device.type == "cuda",
        seed=config.seed,
    )

    model = build_segmentation_model(
        config.training.model,
        class_count=class_count,
        pretrained=args.pretrained,
    ).to(device)
    criterion = BCEDiceLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=3,
    )
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=config.training.amp and device.type == "cuda",
    )

    history: list[dict[str, object]] = []
    best_dice = -1.0
    epochs_without_improvement = 0
    checkpoint_path = Path(args.checkpoint)
    history_path = Path(args.history)

    print("device:", device)
    print("image size:", image_size)
    print("batch size:", batch_size)
    print("train/val samples:", len(train_dataset), len(val_dataset))
    print("AMP:", scaler.is_enabled())

    for epoch in range(1, epochs + 1):
        train_results = run_epoch(
            model=model,
            batches=train_loader,
            criterion=criterion,
            device=device,
            class_count=class_count,
            threshold=config.training.threshold,
            optimizer=optimizer,
            scaler=scaler,
            amp=config.training.amp,
            max_batches=args.max_train_batches,
            description=f"train {epoch}/{epochs}",
        )
        val_results = run_epoch(
            model=model,
            batches=val_loader,
            criterion=criterion,
            device=device,
            class_count=class_count,
            threshold=config.training.threshold,
            amp=config.training.amp,
            max_batches=args.max_val_batches,
            description=f"val {epoch}/{epochs}",
        )
        val_dice = float(val_results["macro_dice"])
        scheduler.step(val_dice)
        epoch_record = {
            "epoch": epoch,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "train": train_results,
            "val": val_results,
        }
        history.append(epoch_record)
        write_json(history_path, history)

        improved = val_dice > best_dice
        if improved:
            best_dice = val_dice
            epochs_without_improvement = 0
            save_checkpoint(
                checkpoint_path,
                {
                    "epoch": epoch,
                    "best_macro_dice": best_dice,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "config": asdict(config),
                },
            )
        else:
            epochs_without_improvement += 1

        print(
            f"epoch={epoch:03d} "
            f"train_loss={float(train_results['loss']):.4f} "
            f"val_loss={float(val_results['loss']):.4f} "
            f"val_macro_dice={val_dice:.4f} "
            f"best={best_dice:.4f} "
            f"train_s={float(train_results['elapsed_seconds']):.1f} "
            f"val_s={float(val_results['elapsed_seconds']):.1f}"
        )
        if epochs_without_improvement >= config.training.patience:
            print(f"early stopping after {epoch} epochs")
            break

    print("best checkpoint:", checkpoint_path)
    print("history:", history_path)


if __name__ == "__main__":
    main()
