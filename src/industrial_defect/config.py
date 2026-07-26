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
class TrainingConfig:
    model: str
    image_size: tuple[int, int]
    epochs: int
    batch: int
    learning_rate: float
    weight_decay: float
    num_workers: int
    amp: bool
    threshold: float
    patience: int
    loss: tuple[str, ...]


@dataclass(frozen=True)
class DeploymentConfig:
    checkpoint: str
    onnx_model: str
    onnx_opset: int
    threshold: float
    input_name: str
    output_name: str
    dynamic_batch: bool


@dataclass(frozen=True)
class ProjectConfig:
    name: str
    seed: int
    dataset: DatasetConfig
    training: TrainingConfig
    deployment: DeploymentConfig


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
    training_raw = _require(raw, "training")
    deployment_raw = _require(raw, "deployment")
    if not all(
        isinstance(section, dict)
        for section in (project_raw, dataset_raw, training_raw, deployment_raw)
    ):
        raise ValueError("project, dataset, training, and deployment must be mappings")

    splits = tuple(_require(dataset_raw, "splits"))
    split_ratio = tuple(float(value) for value in _require(dataset_raw, "split_ratio"))
    classes = tuple(_require(dataset_raw, "classes"))
    image_size = tuple(int(value) for value in _require(dataset_raw, "expected_image_size"))
    training_image_size = tuple(
        int(value) for value in _require(training_raw, "image_size")
    )

    if len(splits) != len(split_ratio):
        raise ValueError("splits and split_ratio must have the same length")
    if abs(sum(split_ratio) - 1.0) > 1e-6:
        raise ValueError("split_ratio must sum to 1.0")
    if len(image_size) != 2:
        raise ValueError("expected_image_size must contain width and height")
    if len(training_image_size) != 2 or any(value <= 0 for value in training_image_size):
        raise ValueError("training.image_size must contain positive width and height")
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
    batch = _require(training_raw, "batch")
    if not isinstance(batch, int) or batch <= 0:
        raise ValueError("training.batch must be a positive integer")

    training = TrainingConfig(
        model=str(_require(training_raw, "model")),
        image_size=(training_image_size[0], training_image_size[1]),
        epochs=int(_require(training_raw, "epochs")),
        batch=batch,
        learning_rate=float(_require(training_raw, "learning_rate")),
        weight_decay=float(_require(training_raw, "weight_decay")),
        num_workers=int(_require(training_raw, "num_workers")),
        amp=bool(_require(training_raw, "amp")),
        threshold=float(_require(training_raw, "threshold")),
        patience=int(_require(training_raw, "patience")),
        loss=tuple(str(value) for value in _require(training_raw, "loss")),
    )
    if training.epochs <= 0 or training.patience <= 0:
        raise ValueError("training epochs and patience must be positive")
    if training.learning_rate <= 0 or training.weight_decay < 0:
        raise ValueError("training optimizer values are invalid")
    if training.num_workers < 0:
        raise ValueError("training.num_workers must be non-negative")
    if not 0.0 < training.threshold < 1.0:
        raise ValueError("training.threshold must be between 0 and 1")
    if not training.loss:
        raise ValueError("training.loss must be non-empty")

    deployment = DeploymentConfig(
        checkpoint=str(_require(deployment_raw, "checkpoint")),
        onnx_model=str(_require(deployment_raw, "onnx_model")),
        onnx_opset=int(_require(deployment_raw, "onnx_opset")),
        threshold=float(_require(deployment_raw, "threshold")),
        input_name=str(_require(deployment_raw, "input_name")),
        output_name=str(_require(deployment_raw, "output_name")),
        dynamic_batch=bool(_require(deployment_raw, "dynamic_batch")),
    )
    if deployment.onnx_opset < 13:
        raise ValueError("deployment.onnx_opset must be at least 13")
    if not 0.0 < deployment.threshold < 1.0:
        raise ValueError("deployment.threshold must be between 0 and 1")
    if not deployment.input_name or not deployment.output_name:
        raise ValueError("deployment input and output names must be non-empty")
    if deployment.input_name == deployment.output_name:
        raise ValueError("deployment input and output names must differ")
    if Path(deployment.onnx_model).suffix.lower() != ".onnx":
        raise ValueError("deployment.onnx_model must use the .onnx suffix")

    return ProjectConfig(
        name=str(_require(project_raw, "name")),
        seed=int(_require(project_raw, "seed")),
        dataset=dataset,
        training=training,
        deployment=deployment,
    )
