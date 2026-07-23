from __future__ import annotations

import csv
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

from industrial_defect.config import DatasetConfig


@dataclass
class DatasetReport:
    root: str
    annotation_rows: int
    images_in_csv: int
    images_on_disk: int
    positive_masks: int
    normal_images: int
    class_masks: dict[str, int]
    missing_images: list[str]
    unexpected_images: list[str]
    invalid_rows: list[str]
    valid: bool
    errors: list[str]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _parse_annotation_row(
    row: dict[str, str], line_number: int
) -> tuple[str, int, str] | str:
    if "ImageId_ClassId" in row:
        combined = (row.get("ImageId_ClassId") or "").strip()
        try:
            image_id, class_id_text = combined.rsplit("_", 1)
            class_id = int(class_id_text)
        except (ValueError, AttributeError):
            return f"train.csv:{line_number}: invalid ImageId_ClassId value"
    elif "ImageId" in row and "ClassId" in row:
        image_id = (row.get("ImageId") or "").strip()
        try:
            class_id = int((row.get("ClassId") or "").strip())
        except ValueError:
            return f"train.csv:{line_number}: invalid ClassId value"
    else:
        return f"train.csv:{line_number}: unsupported annotation columns"

    if not image_id:
        return f"train.csv:{line_number}: image id is empty"

    encoded_pixels = (row.get("EncodedPixels") or "").strip()
    return image_id, class_id, encoded_pixels


def _validate_rle(encoded_pixels: str, line_number: int) -> str | None:
    if not encoded_pixels:
        return None

    tokens = encoded_pixels.split()
    if len(tokens) % 2 != 0:
        return f"train.csv:{line_number}: RLE must contain start-length pairs"

    try:
        values = [int(token) for token in tokens]
    except ValueError:
        return f"train.csv:{line_number}: RLE values must be integers"

    if any(value <= 0 for value in values):
        return f"train.csv:{line_number}: RLE values must be positive"
    return None


def inspect_severstal_dataset(root: str | Path, config: DatasetConfig) -> DatasetReport:
    dataset_root = Path(root).expanduser().resolve()
    csv_path = dataset_root / "train.csv"
    images_dir = dataset_root / "train_images"
    errors: list[str] = []
    invalid_rows: list[str] = []

    if not dataset_root.is_dir():
        errors.append(f"dataset root does not exist: {dataset_root}")
    if not csv_path.is_file():
        errors.append(f"annotation file does not exist: {csv_path}")
    if not images_dir.is_dir():
        errors.append(f"image directory does not exist: {images_dir}")

    if errors:
        return DatasetReport(
            root=str(dataset_root),
            annotation_rows=0,
            images_in_csv=0,
            images_on_disk=0,
            positive_masks=0,
            normal_images=0,
            class_masks={},
            missing_images=[],
            unexpected_images=[],
            invalid_rows=[],
            valid=False,
            errors=errors,
        )

    class_masks: Counter[str] = Counter()
    positive_classes: defaultdict[str, set[int]] = defaultdict(set)
    csv_images: set[str] = set()
    annotation_rows = 0

    with csv_path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = set(reader.fieldnames or [])
        combined_schema = {"ImageId_ClassId", "EncodedPixels"}
        separate_schema = {"ImageId", "ClassId", "EncodedPixels"}
        if not (combined_schema <= fieldnames or separate_schema <= fieldnames):
            errors.append(
                "train.csv must contain ImageId_ClassId and EncodedPixels, "
                "or ImageId, ClassId, and EncodedPixels"
            )
        else:
            for line_number, row in enumerate(reader, start=2):
                annotation_rows += 1
                parsed = _parse_annotation_row(row, line_number)
                if isinstance(parsed, str):
                    invalid_rows.append(parsed)
                    continue

                image_id, class_id, encoded_pixels = parsed
                csv_images.add(image_id)
                if not 1 <= class_id <= len(config.classes):
                    invalid_rows.append(
                        f"train.csv:{line_number}: class id {class_id} is out of range"
                    )
                    continue

                rle_error = _validate_rle(encoded_pixels, line_number)
                if rle_error:
                    invalid_rows.append(rle_error)
                    continue

                if encoded_pixels:
                    class_name = config.classes[class_id - 1]
                    class_masks[class_name] += 1
                    positive_classes[image_id].add(class_id)

    disk_images = {
        path.name
        for path in images_dir.iterdir()
        if path.is_file() and path.suffix.lower() in {".jpeg", ".jpg", ".png"}
    }
    missing_images = sorted(csv_images - disk_images)
    unexpected_images = sorted(disk_images - csv_images)
    positive_masks = sum(class_masks.values())
    normal_images = len(csv_images) - len(positive_classes)

    if annotation_rows != config.expected_annotation_rows:
        errors.append(
            f"expected {config.expected_annotation_rows} annotation rows, "
            f"found {annotation_rows}"
        )
    if len(csv_images) != config.expected_images:
        errors.append(
            f"expected {config.expected_images} images in train.csv, found {len(csv_images)}"
        )
    if len(disk_images) != config.expected_images:
        errors.append(
            f"expected {config.expected_images} image files, found {len(disk_images)}"
        )
    if missing_images:
        errors.append(f"{len(missing_images)} CSV images are missing on disk")
    if unexpected_images:
        errors.append(f"{len(unexpected_images)} image files are absent from train.csv")
    if invalid_rows:
        errors.append(f"{len(invalid_rows)} annotation rows are invalid")

    return DatasetReport(
        root=str(dataset_root),
        annotation_rows=annotation_rows,
        images_in_csv=len(csv_images),
        images_on_disk=len(disk_images),
        positive_masks=positive_masks,
        normal_images=normal_images,
        class_masks=dict(sorted(class_masks.items())),
        missing_images=missing_images,
        unexpected_images=unexpected_images,
        invalid_rows=invalid_rows,
        valid=not errors,
        errors=errors,
    )
