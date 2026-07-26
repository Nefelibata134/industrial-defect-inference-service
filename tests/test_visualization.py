from __future__ import annotations

import numpy as np
import pytest
import torch

from industrial_defect.visualization import (
    FALSE_NEGATIVE_COLOR,
    FALSE_POSITIVE_COLOR,
    TRUE_POSITIVE_COLOR,
    blend_mask,
    build_error_overlay,
    denormalize_image,
)


def test_denormalize_image_restores_imagenet_mean() -> None:
    image = torch.zeros((3, 2, 3), dtype=torch.float32)

    restored = denormalize_image(image)

    assert restored.shape == (2, 3, 3)
    assert restored.dtype == np.uint8
    assert restored[0, 0].tolist() == [124, 116, 104]


def test_build_error_overlay_assigns_tp_fp_and_fn_colors() -> None:
    image = np.zeros((2, 2, 3), dtype=np.uint8)
    target = np.asarray([[1, 0], [1, 0]], dtype=np.uint8)
    prediction = np.asarray([[1, 1], [0, 0]], dtype=np.uint8)

    overlay = build_error_overlay(image, target, prediction, alpha=1.0)

    assert tuple(overlay[0, 0]) == TRUE_POSITIVE_COLOR
    assert tuple(overlay[0, 1]) == FALSE_POSITIVE_COLOR
    assert tuple(overlay[1, 0]) == FALSE_NEGATIVE_COLOR
    assert tuple(overlay[1, 1]) == (0, 0, 0)


def test_blend_mask_rejects_mismatched_shape() -> None:
    image = np.zeros((2, 2, 3), dtype=np.uint8)
    mask = np.zeros((2, 3), dtype=np.uint8)

    with pytest.raises(ValueError, match="mask shape"):
        blend_mask(image, mask, (255, 0, 0), alpha=0.5)
