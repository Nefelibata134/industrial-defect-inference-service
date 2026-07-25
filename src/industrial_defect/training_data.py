from __future__ import annotations

import csv
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path
from typing import TypedDict

import cv2
import numpy as np
import torch
from numpy.typing import NDArray
from torch.utils.data import DataLoader, Dataset

from industrial_defect.rle import decode_rle


class SegmentationSample(TypedDict):
    image: torch.Tensor
    mask: torch.Tensor
    image_id: str


ArrayTransform = Callable[
    [NDArray[np.uint8], NDArray[np.uint8]],
    tuple[NDArray[np.uint8], NDArray[np.uint8]],
]


def read_rle_annotations(
    annotation_path: str | Path,
    class_count: int,
) -> dict[str, tuple[str, ...]]:
    if class_count <= 0:
        raise ValueError("class_count must be positive")

    annotations: defaultdict[str, list[str]] = defaultdict(lambda: [""] * class_count)
    with Path(annotation_path).open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = set(reader.fieldnames or [])
        combined_schema = {"ImageId_ClassId", "EncodedPixels"} <= fieldnames
        separate_schema = {"ImageId", "ClassId", "EncodedPixels"} <= fieldnames
        if not (combined_schema or separate_schema):
            raise ValueError("unsupported annotation columns")

        for line_number, row in enumerate(reader, start=2):
            if combined_schema:
                combined = (row.get("ImageId_ClassId") or "").strip()
                try:
                    image_id, class_id_text = combined.rsplit("_", 1)
                    class_id = int(class_id_text)
                except ValueError as error:
                    raise ValueError(
                        f"train.csv:{line_number}: invalid ImageId_ClassId"
                    ) from error
            else:
                image_id = (row.get("ImageId") or "").strip()
                try:
                    class_id = int((row.get("ClassId") or "").strip())
                except ValueError as error:
                    raise ValueError(f"train.csv:{line_number}: invalid ClassId") from error

            if not image_id:
                raise ValueError(f"train.csv:{line_number}: image id is empty")
            if not 1 <= class_id <= class_count:
                raise ValueError(f"train.csv:{line_number}: class id {class_id} is out of range")
            if annotations[image_id][class_id - 1]:
                raise ValueError(
                    f"train.csv:{line_number}: duplicate class {class_id} for {image_id}"
                )
            annotations[image_id][class_id - 1] = (row.get("EncodedPixels") or "").strip()

    return {image_id: tuple(rles) for image_id, rles in annotations.items()}


def read_manifest_split(manifest_path: str | Path, split: str) -> list[str]:
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
    if len(set(image_ids)) != len(image_ids):
        raise ValueError(f"manifest split contains duplicate image ids: {split}")
    return image_ids


class SeverstalSegmentationDataset(Dataset[SegmentationSample]):
    def __init__(
        self,
        dataset_root: str | Path,
        manifest_path: str | Path,
        split: str,
        class_count: int,
        image_size: tuple[int, int],
        transform: ArrayTransform | None = None,
    ) -> None:
        self.dataset_root = Path(dataset_root)
        self.images_dir = self.dataset_root / "train_images"
        self.image_ids = read_manifest_split(manifest_path, split)
        self.annotations = read_rle_annotations(
            self.dataset_root / "train.csv",
            class_count,
        )
        self.class_count = class_count
        self.image_size = image_size
        self.transform = transform

        if len(image_size) != 2 or any(value <= 0 for value in image_size):
            raise ValueError("image_size must contain positive width and height")
        missing = [
            image_id
            for image_id in self.image_ids
            if not (self.images_dir / image_id).is_file()
        ]
        if missing:
            raise FileNotFoundError(f"{len(missing)} manifest images are missing on disk")

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, index: int) -> SegmentationSample:
        image_id = self.image_ids[index]
        image = cv2.imread(str(self.images_dir / image_id), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"failed to decode image: {image_id}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        height, width = image.shape[:2]
        rles = self.annotations.get(image_id, ("",) * self.class_count)
        masks = np.stack(
            [decode_rle(encoded_pixels, (height, width)) for encoded_pixels in rles],
            axis=-1,
        )

        if self.transform is not None:
            image, masks = self.transform(image, masks)

        target_width, target_height = self.image_size
        if (width, height) != self.image_size:
            image = cv2.resize(
                image,
                (target_width, target_height),
                interpolation=cv2.INTER_LINEAR,
            )
            masks = cv2.resize(
                masks,
                (target_width, target_height),
                interpolation=cv2.INTER_NEAREST,
            )
            if masks.ndim == 2:
                masks = masks[..., None]

        image_array = image.astype(np.float32) / 255.0
        mean = np.asarray((0.485, 0.456, 0.406), dtype=np.float32)
        std = np.asarray((0.229, 0.224, 0.225), dtype=np.float32)
        image_array = (image_array - mean) / std

        image_tensor = torch.from_numpy(np.ascontiguousarray(image_array.transpose(2, 0, 1)))
        mask_tensor = torch.from_numpy(
            np.ascontiguousarray(masks.transpose(2, 0, 1).astype(np.float32))
        )
        return {"image": image_tensor, "mask": mask_tensor, "image_id": image_id}


def build_dataloader(
    dataset: Dataset[SegmentationSample],
    *,
    batch_size: int,
    shuffle: bool,
    num_workers: int = 0,
    pin_memory: bool = False,
    drop_last: bool = False,
    seed: int = 42,
) -> DataLoader:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if num_workers < 0:
        raise ValueError("num_workers must be non-negative")

    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        persistent_workers=num_workers > 0,
        generator=generator,
    )
