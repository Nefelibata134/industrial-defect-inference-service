from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot segmentation training history.")
    parser.add_argument("--history", default="outputs/reports/training_history.json")
    parser.add_argument("--output", default="outputs/reports/training_curves.png")
    return parser.parse_args()


def load_history(path: Path) -> list[dict[str, Any]]:
    history = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(history, list) or not history:
        raise ValueError("training history must be a non-empty list")
    return history


def main() -> None:
    args = parse_args()
    history_path = Path(args.history)
    output_path = Path(args.output)
    history = load_history(history_path)

    epochs = [record["epoch"] for record in history]
    train_loss = [record["train"]["loss"] for record in history]
    val_loss = [record["val"]["loss"] for record in history]
    val_dice = [record["val"]["macro_dice"] for record in history]

    figure, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(epochs, train_loss, marker="o", label="train")
    axes[0].plot(epochs, val_loss, marker="o", label="validation")
    axes[0].set(title="BCE + Dice Loss", xlabel="Epoch", ylabel="Loss")
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    axes[1].plot(epochs, val_dice, marker="o", color="tab:green")
    axes[1].set(title="Validation Macro Dice", xlabel="Epoch", ylabel="Dice", ylim=(0, 1))
    axes[1].grid(alpha=0.3)

    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=160)
    print("saved:", output_path)


if __name__ == "__main__":
    main()
