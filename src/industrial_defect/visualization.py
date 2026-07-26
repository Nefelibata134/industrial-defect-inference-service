from __future__ import annotations

import cv2
import numpy as np
import torch

IMAGENET_MEAN = np.asarray((0.485, 0.456, 0.406), dtype=np.float32)
IMAGENET_STD = np.asarray((0.229, 0.224, 0.225), dtype=np.float32)

TARGET_COLOR = (30, 120, 230)
PREDICTION_COLOR = (240, 150, 30)
TRUE_POSITIVE_COLOR = (30, 200, 70)
FALSE_POSITIVE_COLOR = (230, 50, 50)
FALSE_NEGATIVE_COLOR = (40, 110, 240)


def denormalize_image(image: torch.Tensor) -> np.ndarray:
    if image.ndim != 3 or image.shape[0] != 3:
        raise ValueError("image must have shape 3 x height x width")

    image_array = image.detach().cpu().to(torch.float32).numpy().transpose(1, 2, 0)
    image_array = image_array * IMAGENET_STD + IMAGENET_MEAN
    return np.rint(np.clip(image_array, 0.0, 1.0) * 255.0).astype(np.uint8)


def blend_mask(
    image: np.ndarray,
    mask: np.ndarray,
    color: tuple[int, int, int],
    *,
    alpha: float,
) -> np.ndarray:
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("image must have shape height x width x 3")
    if mask.shape != image.shape[:2]:
        raise ValueError("mask shape must match image height and width")
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be between 0 and 1")

    result = image.copy()
    boolean_mask = mask.astype(bool)
    overlay_color = np.asarray(color, dtype=np.float32)
    result[boolean_mask] = np.rint(
        (1.0 - alpha) * result[boolean_mask].astype(np.float32)
        + alpha * overlay_color
    ).astype(np.uint8)
    return result


def build_error_overlay(
    image: np.ndarray,
    target_mask: np.ndarray,
    predicted_mask: np.ndarray,
    *,
    alpha: float = 0.75,
) -> np.ndarray:
    if target_mask.shape != predicted_mask.shape:
        raise ValueError("target and predicted masks must have matching shapes")
    if target_mask.shape != image.shape[:2]:
        raise ValueError("mask shape must match image height and width")

    target = target_mask.astype(bool)
    prediction = predicted_mask.astype(bool)
    result = blend_mask(
        image,
        target & prediction,
        TRUE_POSITIVE_COLOR,
        alpha=alpha,
    )
    result = blend_mask(
        result,
        ~target & prediction,
        FALSE_POSITIVE_COLOR,
        alpha=alpha,
    )
    return blend_mask(
        result,
        target & ~prediction,
        FALSE_NEGATIVE_COLOR,
        alpha=alpha,
    )


def add_title(image: np.ndarray, title: str, subtitle: str = "") -> np.ndarray:
    title_height = 54
    canvas = np.full(
        (image.shape[0] + title_height, image.shape[1], 3),
        28,
        dtype=np.uint8,
    )
    canvas[title_height:] = image
    cv2.putText(
        canvas,
        title,
        (12, 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (245, 245, 245),
        1,
        cv2.LINE_AA,
    )
    if subtitle:
        cv2.putText(
            canvas,
            subtitle,
            (12, 43),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (190, 190, 190),
            1,
            cv2.LINE_AA,
        )
    return canvas


def build_comparison_panel(
    image: np.ndarray,
    target_mask: np.ndarray,
    predicted_mask: np.ndarray,
    *,
    class_name: str,
    summary: str,
    alpha: float = 0.75,
) -> np.ndarray:
    target_overlay = blend_mask(image, target_mask, TARGET_COLOR, alpha=alpha)
    prediction_overlay = blend_mask(
        image,
        predicted_mask,
        PREDICTION_COLOR,
        alpha=alpha,
    )
    error_overlay = build_error_overlay(
        image,
        target_mask,
        predicted_mask,
        alpha=alpha,
    )

    panels = (
        add_title(image, "Original", summary),
        add_title(target_overlay, f"Ground truth: {class_name}", "blue = target"),
        add_title(
            prediction_overlay,
            f"Prediction: {class_name}",
            "orange = prediction",
        ),
        add_title(
            error_overlay,
            "Pixel error map",
            "green = TP | red = FP | blue = FN",
        ),
    )
    top = np.concatenate(panels[:2], axis=1)
    bottom = np.concatenate(panels[2:], axis=1)
    return np.concatenate((top, bottom), axis=0)
