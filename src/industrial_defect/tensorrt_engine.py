from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from industrial_defect.artifacts import sha256_file


def batch_shapes(
    profile_batch: tuple[int, int, int],
    *,
    channels: int,
    height: int,
    width: int,
) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    if len(profile_batch) != 3:
        raise ValueError("profile batch must contain min, opt, and max")
    if any(value <= 0 for value in profile_batch):
        raise ValueError("profile batch values must be positive")
    if list(profile_batch) != sorted(profile_batch):
        raise ValueError("profile batch values must satisfy min <= opt <= max")
    if any(value <= 0 for value in (channels, height, width)):
        raise ValueError("channels, height, and width must be positive")
    min_batch, opt_batch, max_batch = profile_batch
    return (
        (min_batch, channels, height, width),
        (opt_batch, channels, height, width),
        (max_batch, channels, height, width),
    )


def _load_tensorrt() -> Any:
    try:
        import tensorrt as trt
    except ImportError as error:
        raise RuntimeError(
            "TensorRT is unavailable; install the project tensorrt extra "
            "inside a CUDA-enabled Linux environment"
        ) from error
    return trt


def _parser_errors(parser: Any) -> list[str]:
    return [str(parser.get_error(index)) for index in range(parser.num_errors)]


def _engine_summary(engine: Any, trt: Any) -> dict[str, Any]:
    tensors: list[dict[str, Any]] = []
    for index in range(engine.num_io_tensors):
        name = engine.get_tensor_name(index)
        mode = engine.get_tensor_mode(name)
        tensor = {
            "name": name,
            "mode": str(mode).split(".")[-1].lower(),
            "dtype": str(engine.get_tensor_dtype(name)),
            "shape": list(engine.get_tensor_shape(name)),
        }
        if mode == trt.TensorIOMode.INPUT:
            tensor["profile"] = [
                list(shape) for shape in engine.get_tensor_profile_shape(name, 0)
            ]
        tensors.append(tensor)
    return {
        "num_layers": engine.num_layers,
        "num_optimization_profiles": engine.num_optimization_profiles,
        "device_memory_bytes": engine.device_memory_size_v2,
        "io_tensors": tensors,
    }


def build_tensorrt_engine(
    onnx_path: str | Path,
    engine_path: str | Path,
    *,
    precision: str,
    input_name: str,
    profile_batch: tuple[int, int, int],
    channels: int,
    height: int,
    width: int,
    workspace_gib: int,
) -> dict[str, Any]:
    if precision not in {"fp32", "fp16"}:
        raise ValueError("precision must be fp32 or fp16")
    if workspace_gib <= 0:
        raise ValueError("workspace_gib must be positive")

    source = Path(onnx_path)
    if not source.is_file():
        raise FileNotFoundError(f"ONNX model does not exist: {source}")
    destination = Path(engine_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    min_shape, opt_shape, max_shape = batch_shapes(
        profile_batch,
        channels=channels,
        height=height,
        width=width,
    )

    trt = _load_tensorrt()
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(network_flags)
    parser = trt.OnnxParser(network, logger)
    if not parser.parse(source.read_bytes()):
        details = "\n".join(_parser_errors(parser))
        raise RuntimeError(f"TensorRT could not parse {source}:\n{details}")
    if network.num_inputs != 1 or network.num_outputs != 1:
        raise ValueError("expected exactly one TensorRT input and one output")
    if network.get_input(0).name != input_name:
        raise ValueError(
            f"unexpected TensorRT input name: {network.get_input(0).name}"
        )

    config = builder.create_builder_config()
    config.set_memory_pool_limit(
        trt.MemoryPoolType.WORKSPACE,
        workspace_gib * 1024**3,
    )
    config.clear_flag(trt.BuilderFlag.TF32)
    if precision == "fp16":
        if not builder.platform_has_fast_fp16:
            raise RuntimeError("the current GPU does not report fast FP16 support")
        config.set_flag(trt.BuilderFlag.FP16)

    profile = builder.create_optimization_profile()
    profile.set_shape(input_name, min_shape, opt_shape, max_shape)
    if not profile:
        raise ValueError("TensorRT rejected the optimization profile")
    if config.add_optimization_profile(profile) < 0:
        raise RuntimeError("TensorRT could not add the optimization profile")

    started = time.perf_counter()
    serialized_engine = builder.build_serialized_network(network, config)
    build_seconds = time.perf_counter() - started
    if serialized_engine is None:
        raise RuntimeError("TensorRT engine build failed")

    temporary = destination.with_suffix(f"{destination.suffix}.tmp")
    temporary.write_bytes(bytes(serialized_engine))
    temporary.replace(destination)

    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(destination.read_bytes())
    if engine is None:
        raise RuntimeError("TensorRT could not deserialize the generated engine")

    return {
        "precision": precision,
        "tensorrt_version": trt.__version__,
        "onnx_path": str(source),
        "onnx_sha256": sha256_file(source),
        "engine_path": str(destination),
        "engine_sha256": sha256_file(destination),
        "engine_size_bytes": destination.stat().st_size,
        "build_seconds": build_seconds,
        "workspace_gib": workspace_gib,
        "tf32_enabled": config.get_flag(trt.BuilderFlag.TF32),
        "fp16_enabled": config.get_flag(trt.BuilderFlag.FP16),
        "profile_shapes": {
            "min": list(min_shape),
            "opt": list(opt_shape),
            "max": list(max_shape),
        },
        "engine": _engine_summary(engine, trt),
    }
