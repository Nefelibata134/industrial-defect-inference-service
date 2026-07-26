from __future__ import annotations

from typing import TypedDict

import torch


class ImageClassStatistics(TypedDict):
    intersection: torch.Tensor
    predicted_pixels: torch.Tensor
    target_pixels: torch.Tensor
    false_positive_pixels: torch.Tensor
    false_negative_pixels: torch.Tensor
    union: torch.Tensor
    dice: torch.Tensor
    iou: torch.Tensor
    precision: torch.Tensor
    recall: torch.Tensor
    target_present: torch.Tensor
    prediction_present: torch.Tensor
    image_true_positive: torch.Tensor
    image_false_positive: torch.Tensor
    image_false_negative: torch.Tensor
    image_true_negative: torch.Tensor
    image_has_overlap: torch.Tensor


class ImagePresenceCounts(TypedDict):
    true_positive: torch.Tensor
    false_positive: torch.Tensor
    false_negative: torch.Tensor
    true_negative: torch.Tensor
    has_overlap: torch.Tensor


def _safe_ratio(numerator: torch.Tensor, denominator: torch.Tensor) -> torch.Tensor:
    result = torch.full(
        denominator.shape,
        torch.nan,
        dtype=torch.float64,
        device=denominator.device,
    )
    valid = denominator > 0
    result[valid] = numerator[valid].to(torch.float64) / denominator[valid].to(
        torch.float64
    )
    return result


@torch.no_grad()
def compute_image_class_statistics(
    predicted_masks: torch.Tensor,
    target_masks: torch.Tensor,
) -> ImageClassStatistics:
    """Compute pixel and presence statistics for every image and class."""
    if predicted_masks.shape != target_masks.shape:
        raise ValueError("predicted and target masks must have the same shape")
    if predicted_masks.ndim != 4:
        raise ValueError("segmentation masks must have shape NCHW")
    if predicted_masks.dtype != torch.bool or target_masks.dtype != torch.bool:
        raise TypeError("predicted and target masks must use torch.bool")

    reduction_dims = (2, 3)
    intersection = (predicted_masks & target_masks).sum(dim=reduction_dims)
    predicted_pixels = predicted_masks.sum(dim=reduction_dims)
    target_pixels = target_masks.sum(dim=reduction_dims)
    false_positive_pixels = (predicted_masks & ~target_masks).sum(dim=reduction_dims)
    false_negative_pixels = (~predicted_masks & target_masks).sum(dim=reduction_dims)
    union = (predicted_masks | target_masks).sum(dim=reduction_dims)

    dice = _safe_ratio(2 * intersection, predicted_pixels + target_pixels)
    iou = _safe_ratio(intersection, union)
    precision = _safe_ratio(intersection, predicted_pixels)
    recall = _safe_ratio(intersection, target_pixels)

    target_present = target_pixels > 0
    prediction_present = predicted_pixels > 0

    return {
        "intersection": intersection,
        "predicted_pixels": predicted_pixels,
        "target_pixels": target_pixels,
        "false_positive_pixels": false_positive_pixels,
        "false_negative_pixels": false_negative_pixels,
        "union": union,
        "dice": dice,
        "iou": iou,
        "precision": precision,
        "recall": recall,
        "target_present": target_present,
        "prediction_present": prediction_present,
        "image_true_positive": target_present & prediction_present,
        "image_false_positive": ~target_present & prediction_present,
        "image_false_negative": target_present & ~prediction_present,
        "image_true_negative": ~target_present & ~prediction_present,
        "image_has_overlap": intersection > 0,
    }


@torch.no_grad()
def aggregate_image_presence(
    statistics: ImageClassStatistics,
) -> ImagePresenceCounts:
    """Aggregate per-image presence outcomes while preserving the class axis."""
    return {
        "true_positive": statistics["image_true_positive"].sum(dim=0),
        "false_positive": statistics["image_false_positive"].sum(dim=0),
        "false_negative": statistics["image_false_negative"].sum(dim=0),
        "true_negative": statistics["image_true_negative"].sum(dim=0),
        "has_overlap": statistics["image_has_overlap"].sum(dim=0),
    }
