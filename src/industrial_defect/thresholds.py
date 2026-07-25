from __future__ import annotations

from collections.abc import Sequence

import torch


class ThresholdSweepMetrics:
    def __init__(self, class_count: int, thresholds: Sequence[float]) -> None:
        if class_count <= 0:
            raise ValueError("class_count must be positive")

        normalized = tuple(sorted(float(threshold) for threshold in thresholds))
        if not normalized:
            raise ValueError("at least one threshold is required")
        if len(set(normalized)) != len(normalized):
            raise ValueError("thresholds must be unique")
        if any(not 0.0 < threshold < 1.0 for threshold in normalized):
            raise ValueError("thresholds must be between 0 and 1")

        self.class_count = class_count
        self.thresholds = normalized
        shape = (len(normalized), class_count)
        self.intersection = torch.zeros(shape, dtype=torch.float64)
        self.predicted_pixels = torch.zeros(shape, dtype=torch.float64)
        self.target_pixels = torch.zeros(class_count, dtype=torch.float64)

    @torch.no_grad()
    def update(self, logits: torch.Tensor, targets: torch.Tensor) -> None:
        if logits.shape != targets.shape:
            raise ValueError("logits and targets must have the same shape")
        if logits.ndim != 4 or logits.shape[1] != self.class_count:
            raise ValueError("segmentation tensors must have shape NCHW")

        probabilities = torch.sigmoid(logits)
        target_masks = targets >= 0.5
        reduction_dims = (0, 2, 3)
        self.target_pixels += (
            target_masks.sum(dim=reduction_dims).detach().cpu().to(torch.float64)
        )

        for index, threshold in enumerate(self.thresholds):
            predictions = probabilities >= threshold
            self.intersection[index] += (
                (predictions & target_masks)
                .sum(dim=reduction_dims)
                .detach()
                .cpu()
                .to(torch.float64)
            )
            self.predicted_pixels[index] += (
                predictions.sum(dim=reduction_dims).detach().cpu().to(torch.float64)
            )

    def compute(self) -> list[dict[str, float | list[float]]]:
        results: list[dict[str, float | list[float]]] = []
        for index, threshold in enumerate(self.thresholds):
            intersection = self.intersection[index]
            predicted_pixels = self.predicted_pixels[index]
            dice_denominator = predicted_pixels + self.target_pixels
            dice = torch.where(
                dice_denominator > 0,
                2.0 * intersection / dice_denominator,
                torch.ones_like(dice_denominator),
            )
            precision = torch.where(
                predicted_pixels > 0,
                intersection / predicted_pixels,
                torch.ones_like(predicted_pixels),
            )
            recall = torch.where(
                self.target_pixels > 0,
                intersection / self.target_pixels,
                torch.ones_like(self.target_pixels),
            )
            results.append(
                {
                    "threshold": threshold,
                    "per_class_dice": dice.tolist(),
                    "macro_dice": float(dice.mean()),
                    "per_class_precision": precision.tolist(),
                    "per_class_recall": recall.tolist(),
                }
            )
        return results


def select_best_thresholds(
    results: Sequence[dict[str, float | list[float]]],
) -> list[dict[str, float | int]]:
    if not results:
        raise ValueError("threshold results must not be empty")

    first_scores = results[0].get("per_class_dice")
    if not isinstance(first_scores, list) or not first_scores:
        raise ValueError("threshold results must contain per-class Dice scores")

    best: list[dict[str, float | int]] = []
    for class_index in range(len(first_scores)):
        candidates: list[tuple[float, float]] = []
        for result in results:
            scores = result.get("per_class_dice")
            threshold = result.get("threshold")
            if not isinstance(scores, list) or len(scores) != len(first_scores):
                raise ValueError("inconsistent per-class Dice result shape")
            if not isinstance(threshold, int | float):
                raise ValueError("threshold result is missing a numeric threshold")
            candidates.append((float(scores[class_index]), float(threshold)))

        dice, threshold = max(candidates)
        best.append(
            {
                "class_id": class_index + 1,
                "threshold": threshold,
                "dice": dice,
            }
        )
    return best
