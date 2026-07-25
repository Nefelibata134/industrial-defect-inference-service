from __future__ import annotations

import torch


class SegmentationMetrics:
    def __init__(self, class_count: int, *, threshold: float = 0.5) -> None:
        if class_count <= 0:
            raise ValueError("class_count must be positive")
        if not 0.0 < threshold < 1.0:
            raise ValueError("threshold must be between 0 and 1")

        self.class_count = class_count
        self.threshold = threshold
        self.intersection = torch.zeros(class_count, dtype=torch.float64)
        self.predicted_pixels = torch.zeros(class_count, dtype=torch.float64)
        self.target_pixels = torch.zeros(class_count, dtype=torch.float64)
        self.union = torch.zeros(class_count, dtype=torch.float64)
        self.defect_images = 0
        self.missed_defect_images = 0

    @torch.no_grad()
    def update(self, logits: torch.Tensor, targets: torch.Tensor) -> None:
        if logits.shape != targets.shape:
            raise ValueError("logits and targets must have the same shape")
        if logits.ndim != 4 or logits.shape[1] != self.class_count:
            raise ValueError("segmentation tensors must have shape NCHW")

        predictions = torch.sigmoid(logits) >= self.threshold
        target_masks = targets >= 0.5
        reduction_dims = (0, 2, 3)

        intersection = (predictions & target_masks).sum(dim=reduction_dims)
        predicted_pixels = predictions.sum(dim=reduction_dims)
        target_pixels = target_masks.sum(dim=reduction_dims)
        union = (predictions | target_masks).sum(dim=reduction_dims)

        self.intersection += intersection.detach().cpu().to(torch.float64)
        self.predicted_pixels += predicted_pixels.detach().cpu().to(torch.float64)
        self.target_pixels += target_pixels.detach().cpu().to(torch.float64)
        self.union += union.detach().cpu().to(torch.float64)

        target_positive = target_masks.flatten(1).any(dim=1)
        prediction_positive = predictions.flatten(1).any(dim=1)
        self.defect_images += int(target_positive.sum().item())
        self.missed_defect_images += int((target_positive & ~prediction_positive).sum().item())

    def compute(self) -> dict[str, float | list[float]]:
        dice_denominator = self.predicted_pixels + self.target_pixels
        dice = torch.where(
            dice_denominator > 0,
            2.0 * self.intersection / dice_denominator,
            torch.ones_like(dice_denominator),
        )
        iou = torch.where(
            self.union > 0,
            self.intersection / self.union,
            torch.ones_like(self.union),
        )
        precision = torch.where(
            self.predicted_pixels > 0,
            self.intersection / self.predicted_pixels,
            torch.ones_like(self.predicted_pixels),
        )
        recall = torch.where(
            self.target_pixels > 0,
            self.intersection / self.target_pixels,
            torch.ones_like(self.target_pixels),
        )
        image_false_negative_rate = (
            self.missed_defect_images / self.defect_images if self.defect_images else 0.0
        )

        return {
            "per_class_dice": dice.tolist(),
            "macro_dice": float(dice.mean()),
            "per_class_iou": iou.tolist(),
            "macro_iou": float(iou.mean()),
            "per_class_precision": precision.tolist(),
            "macro_precision": float(precision.mean()),
            "per_class_recall": recall.tolist(),
            "macro_recall": float(recall.mean()),
            "image_false_negative_rate": image_false_negative_rate,
        }
