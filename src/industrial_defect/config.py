from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class DatasetConfig:
    name: str
    task: str
    expected_images: int
    expected_annotation_rows: int
    expected_image_size: tuple[int, int]
    splits: tuple[str, ...]
    split_ratio: tuple[float, ...]
    classes: tuple[str, ...]


@dataclass(frozen=True)
class ProjectConfig:
    name: str
    seed: int
    dataset: DatasetConfig


def _require(mapping: dict[str, Any], key: str) -> Any:
    if key not in mapping:
        raise ValueError(f"missing configuration key: {key}")
    return mapping[key]


def load_project_config(path: str | Path) -> ProjectConfig:
    config_path = Path(path)
    with config_path.open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)

    if not isinstance(raw, dict):
        raise ValueError("configuration root must be a mapping")

    project_raw = _require(raw, "project")
    dataset_raw = _require(raw, "dataset")
    if not isinstance(project_raw, dict) or not isinstance(dataset_raw, dict):
        raise ValueError("project and dataset must be mappings")

    splits = tuple(_require(dataset_raw, "splits"))
    split_ratio = tuple(float(value) for value in _require(dataset_raw, "split_ratio"))
    classes = tuple(_require(dataset_raw, "classes"))
    image_size = tuple(int(value) for value in _require(dataset_raw, "expected_image_size"))

    if len(splits) != len(split_ratio):
        raise ValueError("splits and split_ratio must have the same length")
    if abs(sum(split_ratio) - 1.0) > 1e-6:
        raise ValueError("split_ratio must sum to 1.0")
    if len(image_size) != 2:
        raise ValueError("expected_image_size must contain width and height")
    if not classes or len(set(classes)) != len(classes):
        raise ValueError("classes must be non-empty and unique")

    dataset = DatasetConfig(
        name=str(_require(dataset_raw, "name")),
        task=str(_require(dataset_raw, "task")),
        expected_images=int(_require(dataset_raw, "expected_images")),
        expected_annotation_rows=int(_require(dataset_raw, "expected_annotation_rows")),
        expected_image_size=(image_size[0], image_size[1]),
        splits=splits,
        split_ratio=split_ratio,
        classes=classes,
    )
    return ProjectConfig(
        name=str(_require(project_raw, "name")),
        seed=int(_require(project_raw, "seed")),
        dataset=dataset,
    )
