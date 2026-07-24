from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray


def decode_rle(
    encoded_pixels: str | None,
    shape: Sequence[int],
) -> NDArray[np.uint8]:
    """Decode a one-indexed, column-major Kaggle RLE mask."""
    if len(shape) != 2:
        raise ValueError(f"shape must contain height and width, got {tuple(shape)}")

    height, width = (int(value) for value in shape)
    if height <= 0 or width <= 0:
        raise ValueError(f"shape values must be positive, got {(height, width)}")

    mask_size = height * width
    if encoded_pixels is None or not encoded_pixels.strip():
        return np.zeros((height, width), dtype=np.uint8)

    tokens = encoded_pixels.split()
    if len(tokens) % 2 != 0:
        raise ValueError("RLE must contain start-length pairs")

    try:
        runs = np.asarray(tokens, dtype=np.int64).reshape(-1, 2)
    except ValueError as error:
        raise ValueError("RLE values must be integers") from error

    starts = runs[:, 0] - 1
    lengths = runs[:, 1]
    ends = starts + lengths

    if np.any(starts < 0) or np.any(lengths <= 0):
        raise ValueError("RLE starts and lengths must be positive")
    if np.any(ends > mask_size):
        raise ValueError(f"RLE run exceeds mask size {mask_size}")
    if len(starts) > 1:
        if np.any(starts[1:] < starts[:-1]):
            raise ValueError("RLE runs must be sorted by start position")
        if np.any(starts[1:] < ends[:-1]):
            raise ValueError("RLE runs must not overlap")

    flat_mask = np.zeros(mask_size, dtype=np.uint8)
    for start, end in zip(starts, ends, strict=True):
        flat_mask[start:end] = 1

    return flat_mask.reshape((height, width), order="F")


def encode_rle(mask: NDArray[np.generic]) -> str:
    """Encode a 2D binary mask as one-indexed, column-major Kaggle RLE."""
    mask_array = np.asarray(mask)
    if mask_array.ndim != 2:
        raise ValueError(f"mask must be 2D, got shape {mask_array.shape}")

    foreground = np.asarray(mask_array > 0, dtype=np.uint8).reshape(-1, order="F")
    padded = np.pad(foreground, (1, 1), mode="constant")
    transitions = np.flatnonzero(padded[1:] != padded[:-1]) + 1

    if transitions.size == 0:
        return ""

    runs = transitions.reshape(-1, 2)
    runs[:, 1] -= runs[:, 0]
    return " ".join(str(int(value)) for value in runs.ravel())
