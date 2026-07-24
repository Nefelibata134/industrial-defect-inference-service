from __future__ import annotations

import csv
import hashlib
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import TextIO


@dataclass(frozen=True)
class ImageLabels:
    image_id: str
    labels: tuple[int, ...]

    @property
    def has_defect(self) -> int:
        return int(any(self.labels))


@dataclass(frozen=True)
class ManifestRow:
    image_id: str
    split: str
    labels: tuple[int, ...]

    @property
    def has_defect(self) -> int:
        return int(any(self.labels))


def _annotation_schema(fieldnames: list[str] | None) -> str:
    columns = set(fieldnames or [])
    if {"ImageId_ClassId", "EncodedPixels"} <= columns:
        return "combined"
    if {"ImageId", "ClassId", "EncodedPixels"} <= columns:
        return "separate"
    raise ValueError(
        "annotation CSV must contain ImageId_ClassId and EncodedPixels, "
        "or ImageId, ClassId, and EncodedPixels"
    )


def _parse_identity(row: dict[str, str], schema: str, line_number: int) -> tuple[str, int]:
    if schema == "combined":
        combined = (row.get("ImageId_ClassId") or "").strip()
        try:
            image_id, class_id_text = combined.rsplit("_", 1)
            class_id = int(class_id_text)
        except (AttributeError, ValueError) as error:
            raise ValueError(f"train.csv:{line_number}: invalid ImageId_ClassId value") from error
    else:
        image_id = (row.get("ImageId") or "").strip()
        try:
            class_id = int((row.get("ClassId") or "").strip())
        except ValueError as error:
            raise ValueError(f"train.csv:{line_number}: invalid ClassId value") from error

    if not image_id:
        raise ValueError(f"train.csv:{line_number}: image id is empty")
    return image_id, class_id


def read_image_labels(
    annotation_path: str | Path,
    class_count: int,
    image_ids: list[str] | tuple[str, ...] | None = None,
) -> list[ImageLabels]:
    if class_count <= 0:
        raise ValueError("class_count must be positive")

    if image_ids is not None and len(set(image_ids)) != len(image_ids):
        raise ValueError("image_ids must be unique")

    labels_by_image: dict[str, list[int]] = {
        image_id: [0] * class_count for image_id in (image_ids or ())
    }
    classes_by_image: defaultdict[str, set[int]] = defaultdict(set)

    with Path(annotation_path).open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        schema = _annotation_schema(reader.fieldnames)

        for line_number, row in enumerate(reader, start=2):
            image_id, class_id = _parse_identity(row, schema, line_number)
            if not 1 <= class_id <= class_count:
                raise ValueError(f"train.csv:{line_number}: class id {class_id} is out of range")
            if image_ids is not None and image_id not in labels_by_image:
                raise ValueError(f"train.csv:{line_number}: {image_id} is absent from image_ids")
            if class_id in classes_by_image[image_id]:
                raise ValueError(
                    f"train.csv:{line_number}: duplicate class {class_id} for {image_id}"
                )

            classes_by_image[image_id].add(class_id)
            labels = labels_by_image.setdefault(image_id, [0] * class_count)
            if (row.get("EncodedPixels") or "").strip():
                labels[class_id - 1] = 1

    return [
        ImageLabels(image_id=image_id, labels=tuple(labels_by_image[image_id]))
        for image_id in sorted(labels_by_image)
    ]


def allocate_split_counts(total: int, ratios: tuple[float, ...]) -> tuple[int, ...]:
    if total < 0:
        raise ValueError("total must be non-negative")
    if not ratios or any(ratio < 0 for ratio in ratios):
        raise ValueError("ratios must be non-empty and non-negative")
    if abs(sum(ratios) - 1.0) > 1e-9:
        raise ValueError("ratios must sum to 1.0")

    raw_counts = [total * ratio for ratio in ratios]
    counts = [int(value) for value in raw_counts]
    remainder = total - sum(counts)
    priority = sorted(
        range(len(ratios)),
        key=lambda index: (raw_counts[index] - counts[index], -index),
        reverse=True,
    )
    for index in priority[:remainder]:
        counts[index] += 1
    return tuple(counts)


def _stable_order_key(image_id: str, seed: int) -> bytes:
    return hashlib.sha256(f"{seed}:{image_id}".encode()).digest()


def build_split_manifest(
    records: list[ImageLabels],
    split_names: tuple[str, ...],
    ratios: tuple[float, ...],
    seed: int,
) -> list[ManifestRow]:
    if not records:
        raise ValueError("records must be non-empty")
    if len(split_names) != len(ratios):
        raise ValueError("split_names and ratios must have the same length")
    if len(set(split_names)) != len(split_names):
        raise ValueError("split_names must be unique")

    label_width = len(records[0].labels)
    if label_width == 0 or any(len(record.labels) != label_width for record in records):
        raise ValueError("all records must have the same non-zero label width")
    image_ids = [record.image_id for record in records]
    if len(set(image_ids)) != len(image_ids):
        raise ValueError("image ids must be unique")

    target_counts = list(allocate_split_counts(len(records), ratios))
    remaining_counts = target_counts.copy()
    groups: defaultdict[tuple[int, ...], list[ImageLabels]] = defaultdict(list)
    for record in records:
        groups[record.labels].append(record)

    assignments: list[ManifestRow] = []
    for labels, group in sorted(groups.items(), key=lambda item: (len(item[1]), item[0])):
        ordered_group = sorted(
            group,
            key=lambda record: _stable_order_key(record.image_id, seed),
        )
        group_targets = [len(group) * ratio for ratio in ratios]
        group_assigned = [0] * len(split_names)

        for record in ordered_group:
            candidates = [
                index for index, remaining in enumerate(remaining_counts) if remaining > 0
            ]
            split_index = max(
                candidates,
                key=lambda index: (
                    group_targets[index] - group_assigned[index],
                    remaining_counts[index] / max(target_counts[index], 1),
                    -index,
                ),
            )
            assignments.append(
                ManifestRow(
                    image_id=record.image_id,
                    split=split_names[split_index],
                    labels=labels,
                )
            )
            group_assigned[split_index] += 1
            remaining_counts[split_index] -= 1

    if any(remaining_counts):
        raise RuntimeError(f"split allocation did not fill all targets: {remaining_counts}")
    return sorted(assignments, key=lambda row: row.image_id)


def write_manifest_csv(
    rows: list[ManifestRow],
    class_names: tuple[str, ...],
    output: TextIO,
) -> None:
    writer = csv.writer(output, lineterminator="\n")
    writer.writerow(["image_id", "split", "has_defect", *class_names])
    for row in rows:
        writer.writerow([row.image_id, row.split, row.has_defect, *row.labels])
