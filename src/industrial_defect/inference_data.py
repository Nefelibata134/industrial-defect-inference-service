from __future__ import annotations

import csv
from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray

IMAGENET_MEAN = np.asarray((0.485, 0.456, 0.406), dtype=np.float32)
IMAGENET_STD = np.asarray((0.229, 0.224, 0.225), dtype=np.float32)


def probability_to_logit_threshold(probability: float) -> float:
    if not 0.0 < probability < 1.0:
        raise ValueError("probability threshold must be between 0 and 1")
    return float(np.log(probability / (1.0 - probability)))


def preprocess_image(
    image: NDArray[np.uint8],
    image_size: tuple[int, int],
) -> NDArray[np.float32]:
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("image must have shape height x width x 3")
    if len(image_size) != 2 or any(value <= 0 for value in image_size):
        raise ValueError("image_size must contain positive width and height")

    target_width, target_height = image_size
    if (image.shape[1], image.shape[0]) != image_size:
        image = cv2.resize(
            image,
            (target_width, target_height),
            interpolation=cv2.INTER_LINEAR,
        )
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_array = image_rgb.astype(np.float32) / 255.0
    image_array = (image_array - IMAGENET_MEAN) / IMAGENET_STD
    return np.ascontiguousarray(image_array.transpose(2, 0, 1))


def read_split_image_ids(
    manifest_path: str | Path,
    split: str,
) -> list[str]:
    image_ids: list[str] = []
    with Path(manifest_path).open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if not {"image_id", "split"} <= set(reader.fieldnames or []):
            raise ValueError("manifest must contain image_id and split columns")
        for row in reader:
            if row["split"] == split:
                image_ids.append(row["image_id"])
    if not image_ids:
        raise ValueError(f"manifest contains no images for split: {split}")
    return image_ids


def load_inference_batch(
    dataset_root: str | Path,
    manifest_path: str | Path,
    *,
    split: str,
    batch_size: int,
    image_size: tuple[int, int],
) -> tuple[NDArray[np.float32], list[str]]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    image_ids = read_split_image_ids(manifest_path, split)[:batch_size]
    if len(image_ids) != batch_size:
        raise ValueError(
            f"split {split} contains only {len(image_ids)} images, "
            f"cannot build batch {batch_size}"
        )

    images_dir = Path(dataset_root) / "train_images"
    tensors: list[NDArray[np.float32]] = []
    for image_id in image_ids:
        image = cv2.imread(str(images_dir / image_id), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"failed to decode image: {image_id}")
        tensors.append(preprocess_image(image, image_size))
    return np.stack(tensors, axis=0), image_ids
