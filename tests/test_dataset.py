import csv
from pathlib import Path

from industrial_defect.config import DatasetConfig
from industrial_defect.dataset import inspect_severstal_dataset


def make_config() -> DatasetConfig:
    return DatasetConfig(
        name="fixture",
        task="semantic_segmentation",
        expected_images=3,
        expected_annotation_rows=12,
        expected_image_size=(1600, 256),
        splits=("train", "val", "test"),
        split_ratio=(1 / 3, 1 / 3, 1 / 3),
        classes=("defect_1", "defect_2", "defect_3", "defect_4"),
    )


def write_fixture(root: Path, invalid_rle: bool = False) -> None:
    images_dir = root / "train_images"
    images_dir.mkdir(parents=True)
    image_ids = [f"sample-{index}.jpg" for index in range(3)]
    for image_id in image_ids:
        (images_dir / image_id).write_bytes(b"fixture")

    with (root / "train.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["ImageId_ClassId", "EncodedPixels"])
        writer.writeheader()
        for image_index, image_id in enumerate(image_ids):
            for class_id in range(1, 5):
                encoded_pixels = ""
                if image_index == 0 and class_id == 1:
                    encoded_pixels = "1 3 10 2"
                if invalid_rle and image_index == 1 and class_id == 2:
                    encoded_pixels = "1 3 10"
                writer.writerow(
                    {
                        "ImageId_ClassId": f"{image_id}_{class_id}",
                        "EncodedPixels": encoded_pixels,
                    }
                )


def test_valid_dataset(tmp_path: Path) -> None:
    write_fixture(tmp_path)

    report = inspect_severstal_dataset(tmp_path, make_config())

    assert report.valid
    assert report.annotation_rows == 12
    assert report.images_in_csv == 3
    assert report.images_on_disk == 3
    assert report.positive_masks == 1
    assert report.normal_images == 2
    assert report.class_masks == {"defect_1": 1}


def test_invalid_rle_is_reported(tmp_path: Path) -> None:
    write_fixture(tmp_path, invalid_rle=True)

    report = inspect_severstal_dataset(tmp_path, make_config())

    assert not report.valid
    assert any("annotation rows are invalid" in error for error in report.errors)


def test_missing_dataset_is_reported(tmp_path: Path) -> None:
    report = inspect_severstal_dataset(tmp_path / "missing", make_config())

    assert not report.valid
    assert any("dataset root does not exist" in error for error in report.errors)
