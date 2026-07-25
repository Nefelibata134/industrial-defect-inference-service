import csv
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

from industrial_defect.rle import encode_rle
from industrial_defect.training_data import (
    SeverstalSegmentationDataset,
    build_dataloader,
    compute_class_aware_sample_weights,
)


def write_training_fixture(root: Path) -> tuple[Path, Path]:
    images_dir = root / "train_images"
    images_dir.mkdir()
    image = np.zeros((3, 4, 3), dtype=np.uint8)
    image[:, :, 2] = 255
    assert cv2.imwrite(str(images_dir / "positive.jpg"), image)
    assert cv2.imwrite(str(images_dir / "normal.jpg"), image)

    mask = np.zeros((3, 4), dtype=np.uint8)
    mask[0:2, 0] = 1
    with (root / "train.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["ImageId", "ClassId", "EncodedPixels"])
        writer.writeheader()
        writer.writerow(
            {
                "ImageId": "positive.jpg",
                "ClassId": "1",
                "EncodedPixels": encode_rle(mask),
            }
        )

    manifest_path = root / "manifest.csv"
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["image_id", "split"])
        writer.writeheader()
        writer.writerow({"image_id": "positive.jpg", "split": "train"})
        writer.writerow({"image_id": "normal.jpg", "split": "val"})
    return manifest_path, mask


def test_dataset_returns_normalized_image_and_multiclass_mask(tmp_path: Path) -> None:
    manifest_path, expected_mask = write_training_fixture(tmp_path)
    dataset = SeverstalSegmentationDataset(
        dataset_root=tmp_path,
        manifest_path=manifest_path,
        split="train",
        class_count=4,
        image_size=(4, 3),
    )

    sample = dataset[0]

    assert sample["image_id"] == "positive.jpg"
    assert sample["image"].shape == (3, 3, 4)
    assert sample["image"].dtype.is_floating_point
    assert sample["mask"].shape == (4, 3, 4)
    np.testing.assert_array_equal(sample["mask"][0].numpy(), expected_mask)
    assert not sample["mask"][1:].any()


def test_sparse_annotation_image_has_zero_mask(tmp_path: Path) -> None:
    manifest_path, _ = write_training_fixture(tmp_path)
    dataset = SeverstalSegmentationDataset(
        dataset_root=tmp_path,
        manifest_path=manifest_path,
        split="val",
        class_count=4,
        image_size=(8, 6),
    )

    sample = dataset[0]

    assert sample["image"].shape == (3, 6, 8)
    assert sample["mask"].shape == (4, 6, 8)
    assert not sample["mask"].any()


def test_empty_split_is_rejected(tmp_path: Path) -> None:
    manifest_path, _ = write_training_fixture(tmp_path)

    with pytest.raises(ValueError, match="no images"):
        SeverstalSegmentationDataset(
            dataset_root=tmp_path,
            manifest_path=manifest_path,
            split="test",
            class_count=4,
            image_size=(4, 3),
        )


def test_dataloader_adds_batch_dimension(tmp_path: Path) -> None:
    manifest_path, _ = write_training_fixture(tmp_path)
    dataset = SeverstalSegmentationDataset(
        dataset_root=tmp_path,
        manifest_path=manifest_path,
        split="train",
        class_count=4,
        image_size=(4, 3),
    )
    loader = build_dataloader(
        dataset,
        batch_size=1,
        shuffle=False,
        seed=123,
    )

    batch = next(iter(loader))

    assert batch["image"].shape == (1, 3, 3, 4)
    assert batch["mask"].shape == (1, 4, 3, 4)
    assert batch["image_id"] == ["positive.jpg"]


def test_dataloader_rejects_invalid_batch_size(tmp_path: Path) -> None:
    manifest_path, _ = write_training_fixture(tmp_path)
    dataset = SeverstalSegmentationDataset(
        dataset_root=tmp_path,
        manifest_path=manifest_path,
        split="train",
        class_count=4,
        image_size=(4, 3),
    )

    with pytest.raises(ValueError, match="batch_size"):
        build_dataloader(dataset, batch_size=0, shuffle=False)


def test_class_aware_weights_prioritize_rare_classes() -> None:
    label_matrix = torch.tensor(
        [
            [1, 0],
            [0, 1],
            [0, 1],
            [0, 1],
            [0, 0],
        ],
        dtype=torch.bool,
    )

    weights = compute_class_aware_sample_weights(label_matrix, power=0.5)

    assert weights.mean().item() == pytest.approx(1.0)
    assert weights[0] > weights[1]
    assert weights[4] > weights[1]


def test_class_aware_weights_reject_invalid_power() -> None:
    label_matrix = torch.tensor([[1, 0]], dtype=torch.bool)

    with pytest.raises(ValueError, match="power"):
        compute_class_aware_sample_weights(label_matrix, power=1.5)
