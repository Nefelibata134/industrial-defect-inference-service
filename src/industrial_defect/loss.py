from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def soft_dice_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    smooth: float = 1.0,
) -> torch.Tensor:
    if logits.shape != targets.shape:
        raise ValueError(
            f"logits and targets must have the same shape: {logits.shape} != {targets.shape}"
        )
    if logits.ndim != 4:
        raise ValueError("segmentation tensors must have shape NCHW")

    probabilities = torch.sigmoid(logits)
    reduction_dims = (0, 2, 3)
    intersection = (probabilities * targets).sum(dim=reduction_dims)
    denominator = probabilities.sum(dim=reduction_dims) + targets.sum(dim=reduction_dims)
    class_dice = (2.0 * intersection + smooth) / (denominator + smooth)
    return 1.0 - class_dice.mean()


class BCEDiceLoss(nn.Module):
    def __init__(
        self,
        *,
        bce_weight: float = 0.5,
        dice_weight: float = 0.5,
    ) -> None:
        super().__init__()
        if bce_weight < 0 or dice_weight < 0 or bce_weight + dice_weight <= 0:
            raise ValueError("loss weights must be non-negative with a positive sum")
        weight_sum = bce_weight + dice_weight
        self.bce_weight = bce_weight / weight_sum
        self.dice_weight = dice_weight / weight_sum

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(logits, targets)
        dice = soft_dice_loss(logits, targets)
        return self.bce_weight * bce + self.dice_weight * dice
