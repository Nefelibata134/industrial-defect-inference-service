from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Protocol

import cv2
import numpy as np
from numpy.typing import NDArray

from industrial_defect.rle import encode_rle

CLASS_NAMES = ("defect_1", "defect_2", "defect_3", "defect_4")


@dataclass(frozen=True)
class ServiceSettings:
    triton_url: str = "triton:8000"
    model_name: str = "steel_defect_segmentation"
    model_version: str = "1"
    source_width: int = 1600
    source_height: int = 256
    input_width: int = 1024
    input_height: int = 256
    threshold: float = 0.80
    max_upload_bytes: int = 8 * 1024 * 1024
    request_timeout_seconds: float = 5.0

    @classmethod
    def from_env(cls) -> ServiceSettings:
        return cls(
            triton_url=os.getenv("TRITON_URL", cls.triton_url),
            model_name=os.getenv("TRITON_MODEL_NAME", cls.model_name),
            model_version=os.getenv("TRITON_MODEL_VERSION", cls.model_version),
            threshold=float(os.getenv("SEGMENTATION_THRESHOLD", cls.threshold)),
            max_upload_bytes=int(os.getenv("MAX_UPLOAD_BYTES", cls.max_upload_bytes)),
            request_timeout_seconds=float(
                os.getenv("MODEL_REQUEST_TIMEOUT_SECONDS", cls.request_timeout_seconds)
            ),
        )

    def __post_init__(self) -> None:
        if not 0.0 < self.threshold < 1.0:
            raise ValueError("threshold must be between 0 and 1")
        if self.max_upload_bytes <= 0:
            raise ValueError("max_upload_bytes must be positive")
        if self.request_timeout_seconds <= 0.0:
            raise ValueError("request_timeout_seconds must be positive")
        dimensions = (
            self.source_width,
            self.source_height,
            self.input_width,
            self.input_height,
        )
        if any(value <= 0 for value in dimensions):
            raise ValueError("image dimensions must be positive")


class AsyncModelClient(Protocol):
    async def ready(self) -> bool: ...

    async def infer(
        self,
        images: NDArray[np.float32],
    ) -> tuple[NDArray[np.float32], float]: ...

    async def close(self) -> None: ...


def decode_image(payload: bytes, settings: ServiceSettings) -> NDArray[np.uint8]:
    if not payload:
        raise ValueError("image payload is empty")
    if len(payload) > settings.max_upload_bytes:
        raise ValueError("image payload exceeds the configured byte limit")

    encoded = np.frombuffer(payload, dtype=np.uint8)
    image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("image payload could not be decoded")

    expected_shape = (settings.source_height, settings.source_width, 3)
    if image.shape != expected_shape:
        raise ValueError(f"image must have shape {expected_shape}, got {image.shape}")
    return image


def sigmoid(logits: NDArray[np.float32]) -> NDArray[np.float32]:
    clipped = np.clip(logits, -80.0, 80.0)
    return np.asarray(1.0 / (1.0 + np.exp(-clipped)), dtype=np.float32)


def connected_components(mask: NDArray[np.uint8]) -> list[dict[str, int]]:
    component_count, _, statistics, _ = cv2.connectedComponentsWithStats(
        mask,
        connectivity=8,
    )
    components: list[dict[str, int]] = []
    for index in range(1, component_count):
        x, y, width, height, pixels = statistics[index].tolist()
        components.append(
            {
                "x": int(x),
                "y": int(y),
                "width": int(width),
                "height": int(height),
                "pixels": int(pixels),
            }
        )
    return components


def serialize_segmentation(
    logits: NDArray[np.float32],
    *,
    threshold: float,
    output_size: tuple[int, int],
    class_names: tuple[str, ...] = CLASS_NAMES,
) -> list[dict[str, object]]:
    if logits.ndim != 4 or logits.shape[0] != 1:
        raise ValueError(f"logits must have shape 1 x C x H x W, got {logits.shape}")
    if logits.shape[1] != len(class_names):
        raise ValueError(
            f"logit channels {logits.shape[1]} do not match {len(class_names)} classes"
        )
    if not 0.0 < threshold < 1.0:
        raise ValueError("threshold must be between 0 and 1")

    output_width, output_height = output_size
    probabilities = sigmoid(logits[0])
    classes: list[dict[str, object]] = []

    for index, class_name in enumerate(class_names):
        probability_map = probabilities[index]
        mask = np.asarray(probability_map >= threshold, dtype=np.uint8)
        if (mask.shape[1], mask.shape[0]) != output_size:
            mask = cv2.resize(
                mask,
                (output_width, output_height),
                interpolation=cv2.INTER_NEAREST,
            )

        defect_pixels = int(mask.sum())
        classes.append(
            {
                "class_id": index + 1,
                "class_name": class_name,
                "detected": defect_pixels > 0,
                "max_probability": round(float(probability_map.max()), 6),
                "defect_pixels": defect_pixels,
                "area_ratio": round(defect_pixels / mask.size, 8),
                "components": connected_components(mask),
                "rle": encode_rle(mask),
            }
        )
    return classes
