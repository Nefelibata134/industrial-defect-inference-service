import csv
from collections import Counter
from pathlib import Path

import pytest

from industrial_defect.manifest import (
    ImageLabels,
    allocate_split_counts,
    build_split_manifest,
    read_image_labels,
)


def write_annotations(path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["ImageId_ClassId", "EncodedPixels"])
        writer.writeheader()
        for image_index in range(5):
            for class_id in range(1, 5):
                writer.writerow(
                    {
                        "ImageId_ClassId": f"image-{image_index}.jpg_{class_id}",
                        "EncodedPixels": ("1 2" if image_index == class_id - 1 else ""),
                    }
                )


def test_read_image_labels_builds_multilabel_rows(tmp_path: Path) -> None:
    annotation_path = tmp_path / "train.csv"
    write_annotations(annotation_path)

    records = read_image_labels(annotation_path, class_count=4)

    assert len(records) == 5
    assert records[0].image_id == "image-0.jpg"
    assert records[0].labels == (1, 0, 0, 0)
    assert records[-1].labels == (0, 0, 0, 0)
    assert records[-1].has_defect == 0


def test_read_image_labels_fills_normal_images_for_sparse_csv(tmp_path: Path) -> None:
    annotation_path = tmp_path / "train.csv"
    with annotation_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["ImageId", "ClassId", "EncodedPixels"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "ImageId": "positive.jpg",
                "ClassId": "3",
                "EncodedPixels": "1 2",
            }
        )

    records = read_image_labels(
        annotation_path,
        class_count=4,
        image_ids=["normal.jpg", "positive.jpg"],
    )

    assert records == [
        ImageLabels("normal.jpg", (0, 0, 0, 0)),
        ImageLabels("positive.jpg", (0, 0, 1, 0)),
    ]


def test_read_image_labels_rejects_unknown_sparse_image(tmp_path: Path) -> None:
    annotation_path = tmp_path / "train.csv"
    with annotation_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["ImageId", "ClassId", "EncodedPixels"],
        )
        writer.writeheader()
        writer.writerow({"ImageId": "unknown.jpg", "ClassId": "1", "EncodedPixels": "1 2"})

    with pytest.raises(ValueError, match="absent from image_ids"):
        read_image_labels(
            annotation_path,
            class_count=4,
            image_ids=["known.jpg"],
        )


def test_read_image_labels_rejects_duplicate_class(tmp_path: Path) -> None:
    annotation_path = tmp_path / "train.csv"
    write_annotations(annotation_path)
    with annotation_path.open("a", encoding="utf-8") as handle:
        handle.write("image-0.jpg_1,\n")

    with pytest.raises(ValueError, match="duplicate class"):
        read_image_labels(annotation_path, class_count=4)


def test_allocate_split_counts_uses_largest_remainder() -> None:
    assert allocate_split_counts(11, (0.7, 0.15, 0.15)) == (8, 2, 1)


def make_records() -> list[ImageLabels]:
    records = []
    signatures = [
        (0, 0, 0, 0),
        (1, 0, 0, 0),
        (0, 1, 0, 0),
        (0, 0, 1, 0),
        (0, 0, 0, 1),
        (1, 0, 1, 0),
    ]
    for index in range(100):
        records.append(
            ImageLabels(
                image_id=f"image-{index:03d}.jpg",
                labels=signatures[index % len(signatures)],
            )
        )
    return records


def test_split_manifest_is_exact_and_deterministic() -> None:
    records = make_records()
    kwargs = {
        "split_names": ("train", "val", "test"),
        "ratios": (0.7, 0.15, 0.15),
        "seed": 42,
    }

    first = build_split_manifest(records, **kwargs)
    second = build_split_manifest(list(reversed(records)), **kwargs)

    assert first == second
    assert len({row.image_id for row in first}) == len(records)
    assert Counter(row.split for row in first) == {
        "train": 70,
        "val": 15,
        "test": 15,
    }


def test_split_manifest_rejects_duplicate_ids() -> None:
    records = [
        ImageLabels("same.jpg", (1, 0)),
        ImageLabels("same.jpg", (0, 1)),
    ]

    with pytest.raises(ValueError, match="image ids must be unique"):
        build_split_manifest(
            records,
            split_names=("train", "val"),
            ratios=(0.8, 0.2),
            seed=42,
        )
