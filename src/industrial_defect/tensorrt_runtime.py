from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray


def _load_dependencies() -> tuple[Any, Any]:
    try:
        import tensorrt as trt
        from cuda.bindings import runtime
    except ImportError as error:
        raise RuntimeError(
            "TensorRT and CUDA Python bindings are required for engine execution"
        ) from error
    return trt, runtime


def _cuda_result(result: tuple[Any, ...], operation: str) -> tuple[Any, ...]:
    error, *values = result
    if int(error) != 0:
        raise RuntimeError(f"{operation} failed with CUDA error {error}")
    return tuple(values)


class TensorRTRunner:
    def __init__(self, engine_path: str | Path) -> None:
        self.trt, self.runtime_api = _load_dependencies()
        source = Path(engine_path)
        if not source.is_file():
            raise FileNotFoundError(f"TensorRT engine does not exist: {source}")

        self.logger = self.trt.Logger(self.trt.Logger.WARNING)
        self.runtime = self.trt.Runtime(self.logger)
        self.engine = self.runtime.deserialize_cuda_engine(source.read_bytes())
        if self.engine is None:
            raise RuntimeError(f"failed to deserialize TensorRT engine: {source}")
        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError("failed to create TensorRT execution context")

        input_names: list[str] = []
        output_names: list[str] = []
        for index in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(index)
            mode = self.engine.get_tensor_mode(name)
            if mode == self.trt.TensorIOMode.INPUT:
                input_names.append(name)
            elif mode == self.trt.TensorIOMode.OUTPUT:
                output_names.append(name)
        if len(input_names) != 1 or len(output_names) != 1:
            raise ValueError("runner expects exactly one input and one output tensor")

        self.input_name = input_names[0]
        self.output_name = output_names[0]
        self.input_dtype = np.dtype(
            self.trt.nptype(self.engine.get_tensor_dtype(self.input_name))
        )
        self.output_dtype = np.dtype(
            self.trt.nptype(self.engine.get_tensor_dtype(self.output_name))
        )
        (self.stream,) = _cuda_result(
            self.runtime_api.cudaStreamCreate(),
            "cudaStreamCreate",
        )
        (self.start_event,) = _cuda_result(
            self.runtime_api.cudaEventCreate(),
            "cudaEventCreate(start)",
        )
        (self.end_event,) = _cuda_result(
            self.runtime_api.cudaEventCreate(),
            "cudaEventCreate(end)",
        )
        self.input_device: int | None = None
        self.output_device: int | None = None
        self.output_host: NDArray[Any] | None = None
        self.current_shape: tuple[int, ...] | None = None

    def _free_buffers(self) -> None:
        if self.input_device is not None:
            _cuda_result(
                self.runtime_api.cudaFree(self.input_device),
                "cudaFree(input)",
            )
            self.input_device = None
        if self.output_device is not None:
            _cuda_result(
                self.runtime_api.cudaFree(self.output_device),
                "cudaFree(output)",
            )
            self.output_device = None
        self.output_host = None
        self.current_shape = None

    def _allocate(self, input_shape: tuple[int, ...]) -> None:
        if self.current_shape == input_shape:
            return
        self._free_buffers()
        if not self.context.set_input_shape(self.input_name, input_shape):
            raise ValueError(f"TensorRT rejected input shape: {input_shape}")

        output_shape = tuple(self.context.get_tensor_shape(self.output_name))
        if any(dimension <= 0 for dimension in output_shape):
            raise RuntimeError(f"TensorRT returned invalid output shape: {output_shape}")
        input_bytes = int(np.prod(input_shape, dtype=np.int64)) * self.input_dtype.itemsize
        output_bytes = (
            int(np.prod(output_shape, dtype=np.int64)) * self.output_dtype.itemsize
        )
        (self.input_device,) = _cuda_result(
            self.runtime_api.cudaMalloc(input_bytes),
            "cudaMalloc(input)",
        )
        (self.output_device,) = _cuda_result(
            self.runtime_api.cudaMalloc(output_bytes),
            "cudaMalloc(output)",
        )
        self.output_host = np.empty(output_shape, dtype=self.output_dtype)
        if not self.context.set_tensor_address(self.input_name, self.input_device):
            raise RuntimeError("failed to bind TensorRT input buffer")
        if not self.context.set_tensor_address(self.output_name, self.output_device):
            raise RuntimeError("failed to bind TensorRT output buffer")
        self.current_shape = input_shape

    def infer(
        self,
        inputs: NDArray[np.floating[Any]],
    ) -> tuple[NDArray[Any], float]:
        input_array = np.ascontiguousarray(inputs, dtype=self.input_dtype)
        self._allocate(tuple(input_array.shape))
        if (
            self.input_device is None
            or self.output_device is None
            or self.output_host is None
        ):
            raise RuntimeError("TensorRT buffers are not allocated")

        _cuda_result(
            self.runtime_api.cudaMemcpyAsync(
                self.input_device,
                input_array.ctypes.data,
                input_array.nbytes,
                self.runtime_api.cudaMemcpyKind.cudaMemcpyHostToDevice,
                self.stream,
            ),
            "cudaMemcpyAsync(H2D)",
        )
        _cuda_result(
            self.runtime_api.cudaEventRecord(self.start_event, self.stream),
            "cudaEventRecord(start)",
        )
        if not self.context.execute_async_v3(int(self.stream)):
            raise RuntimeError("TensorRT execute_async_v3 failed")
        _cuda_result(
            self.runtime_api.cudaEventRecord(self.end_event, self.stream),
            "cudaEventRecord(end)",
        )
        _cuda_result(
            self.runtime_api.cudaMemcpyAsync(
                self.output_host.ctypes.data,
                self.output_device,
                self.output_host.nbytes,
                self.runtime_api.cudaMemcpyKind.cudaMemcpyDeviceToHost,
                self.stream,
            ),
            "cudaMemcpyAsync(D2H)",
        )
        _cuda_result(
            self.runtime_api.cudaStreamSynchronize(self.stream),
            "cudaStreamSynchronize",
        )
        (gpu_ms,) = _cuda_result(
            self.runtime_api.cudaEventElapsedTime(
                self.start_event,
                self.end_event,
            ),
            "cudaEventElapsedTime",
        )
        return self.output_host.copy(), float(gpu_ms)

    def close(self) -> None:
        self._free_buffers()
        if getattr(self, "start_event", None) is not None:
            _cuda_result(
                self.runtime_api.cudaEventDestroy(self.start_event),
                "cudaEventDestroy(start)",
            )
            self.start_event = None
        if getattr(self, "end_event", None) is not None:
            _cuda_result(
                self.runtime_api.cudaEventDestroy(self.end_event),
                "cudaEventDestroy(end)",
            )
            self.end_event = None
        if getattr(self, "stream", None) is not None:
            _cuda_result(
                self.runtime_api.cudaStreamDestroy(self.stream),
                "cudaStreamDestroy",
            )
            self.stream = None

    def __enter__(self) -> TensorRTRunner:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
