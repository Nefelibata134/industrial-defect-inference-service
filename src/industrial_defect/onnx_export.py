from __future__ import annotations

import hashlib
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import onnx
import torch
from onnx import TensorProto
from torch import nn

from industrial_defect.config import ProjectConfig
from industrial_defect.model import build_segmentation_model


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_inference_model(
    config: ProjectConfig,
    checkpoint_path: str | Path,
) -> tuple[nn.Module, dict[str, Any]]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        raise ValueError("checkpoint root must be a mapping")

    required_keys = {"epoch", "best_macro_dice", "model_state_dict"}
    missing_keys = required_keys - checkpoint.keys()
    if missing_keys:
        missing = ", ".join(sorted(missing_keys))
        raise ValueError(f"checkpoint is missing required keys: {missing}")

    state_dict = checkpoint["model_state_dict"]
    if not isinstance(state_dict, Mapping):
        raise ValueError("checkpoint model_state_dict must be a mapping")

    model = build_segmentation_model(
        config.training.model,
        class_count=len(config.dataset.classes),
        pretrained=False,
    )
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model, checkpoint


def export_onnx_model(
    model: nn.Module,
    example_input: torch.Tensor,
    output_path: str | Path,
    *,
    input_name: str,
    output_name: str,
    opset: int,
    dynamic_batch: bool,
    metadata: Mapping[str, str],
) -> dict[str, Any]:
    if example_input.ndim != 4:
        raise ValueError("example input must have shape NCHW")
    if example_input.dtype != torch.float32:
        raise ValueError("example input must use float32")
    if opset < 13:
        raise ValueError("ONNX opset must be at least 13")

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    model.eval()
    dynamic_shapes = ({0: "batch"},) if dynamic_batch else None

    with torch.inference_mode():
        onnx_program = torch.onnx.export(
            model,
            (example_input,),
            input_names=[input_name],
            output_names=[output_name],
            opset_version=opset,
            dynamo=True,
            dynamic_shapes=dynamic_shapes,
            external_data=False,
            optimize=True,
        )
    if onnx_program is None:
        raise RuntimeError("PyTorch ONNX exporter did not return an ONNX program")
    onnx_program.save(destination, external_data=False)

    model_proto = onnx.load(destination)
    del model_proto.metadata_props[:]
    for key, value in sorted(metadata.items()):
        property_entry = model_proto.metadata_props.add()
        property_entry.key = key
        property_entry.value = value
    onnx.checker.check_model(model_proto)
    onnx.save(model_proto, destination)
    onnx.checker.check_model(onnx.load(destination))
    summary = summarize_onnx_model(destination)
    actual_opset = summary["opsets"].get("ai.onnx")
    if actual_opset != opset:
        raise RuntimeError(
            f"requested ONNX opset {opset}, but exporter produced {actual_opset}"
        )
    return summary


def _value_info_summary(value_info: onnx.ValueInfoProto) -> dict[str, Any]:
    tensor_type = value_info.type.tensor_type
    shape: list[int | str] = []
    for dimension in tensor_type.shape.dim:
        if dimension.HasField("dim_param") and dimension.dim_param:
            shape.append(dimension.dim_param)
        else:
            shape.append(int(dimension.dim_value))
    return {
        "name": value_info.name,
        "shape": shape,
        "dtype": TensorProto.DataType.Name(tensor_type.elem_type).lower(),
    }


def summarize_onnx_model(path: str | Path) -> dict[str, Any]:
    model_proto = onnx.load(path)
    onnx.checker.check_model(model_proto)
    metadata = {entry.key: entry.value for entry in model_proto.metadata_props}
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "ir_version": model_proto.ir_version,
        "opsets": {
            item.domain or "ai.onnx": item.version for item in model_proto.opset_import
        },
        "inputs": [_value_info_summary(item) for item in model_proto.graph.input],
        "outputs": [_value_info_summary(item) for item in model_proto.graph.output],
        "nodes": len(model_proto.graph.node),
        "initializers": len(model_proto.graph.initializer),
        "metadata": metadata,
    }
