# TensorRT Build and Benchmark

## Objective

Compile the promoted segmentation graph into TensorRT FP32 and FP16 engines,
verify output behavior against ONNX Runtime, and measure deployment
performance under a declared timing boundary.

## Environment

| Item | Value |
| --- | --- |
| Host | WSL2 Ubuntu 22.04, x86_64 |
| GPU | NVIDIA GeForce RTX 4070 Laptop GPU, 8,188 MiB |
| Driver | 610.47 |
| Python | 3.10.20 |
| PyTorch | 2.12.1+cu130 |
| ONNX Runtime GPU | 1.23.2, CUDAExecutionProvider |
| TensorRT | 10.9.0.34, CUDA 12 package |

The PyTorch environment uses its CUDA 13 runtime, while ONNX Runtime and
TensorRT use isolated CUDA 12 environments. The table is an operational
deployment comparison on the same hardware, not a claim that runtime versions
are perfectly controlled. TensorRT FP32-to-FP16 comparisons use the same
TensorRT environment and are directly comparable.

## Engine Contract

| Tensor | Data type | Shape |
| --- | --- | --- |
| `images` | float32 | `[batch, 3, 256, 1024]` |
| `logits` | float32 | `[batch, 4, 256, 1024]` |

The optimization profile is:

```text
minimum: [1, 3, 256, 1024]
optimum: [4, 3, 256, 1024]
maximum: [8, 3, 256, 1024]
```

FP32 explicitly disables TF32. FP16 enables mixed precision for eligible
layers while retaining float32 input and output bindings. Unsupported FP16
layers may execute in FP32.

## Build Artifacts

| Precision | Build time | Engine size | Context memory | SHA256 |
| --- | ---: | ---: | ---: | --- |
| FP32 | 12.55 s | 94.40 MiB | 524 MiB | `423e92182d30e232ab6c85c11145bfeb27c70b48cf9aaf604eb5e6b9cf433c7f` |
| FP16 | 29.25 s | 30.04 MiB | 262 MiB | `0e040ac97a207baf476a07716c16e4c9c024d85d286ac1977e7be8b127ce201d` |

Both engines were generated from ONNX SHA256
`6622d3aada8f7fe0547c475f80a8697cb74f7e6e8c6b677a9dbaff45a4f03a0c`.
The serialized plans are generated release artifacts and are excluded from
source control.

## Protocol

- Input: deterministic validation images from the versioned manifest.
- Input size: `N x 3 x 256 x 1024`.
- Batch sizes: `1`, `4`, and `8`.
- Warmup: 50 runs per backend and batch.
- Measurement: 500 runs per backend and batch.
- Cross-backend boundary: normalized CPU float32 tensor input through CPU
  float32 logits output.
- Excluded: image decoding, resizing, normalization, sigmoid, thresholding,
  and service transport.
- TensorRT additionally records CUDA-event model execution time separately
  from host-to-device and device-to-host transfers.

The PyTorch, ONNX Runtime, and TensorRT benchmarks run in separate pinned
environments to prevent incompatible CUDA dependency stacks from being mixed.

## Performance

Cross-backend tensor I/O latency:

| Backend | Precision | Batch | Mean | P95 | Throughput |
| --- | --- | ---: | ---: | ---: | ---: |
| PyTorch CUDA | FP32 | 1 | 7.02 ms | 7.48 ms | 142.4 images/s |
| PyTorch CUDA | FP32 | 4 | 33.16 ms | 36.87 ms | 120.6 images/s |
| PyTorch CUDA | FP32 | 8 | 68.13 ms | 74.47 ms | 117.4 images/s |
| ONNX Runtime CUDA | FP32 | 1 | 8.23 ms | 9.13 ms | 121.5 images/s |
| ONNX Runtime CUDA | FP32 | 4 | 33.89 ms | 36.66 ms | 118.0 images/s |
| ONNX Runtime CUDA | FP32 | 8 | 70.96 ms | 76.11 ms | 112.7 images/s |
| TensorRT | FP32 | 1 | 5.18 ms | 5.63 ms | 192.9 images/s |
| TensorRT | FP32 | 4 | 20.58 ms | 22.12 ms | 194.4 images/s |
| TensorRT | FP32 | 8 | 44.88 ms | 49.51 ms | 178.2 images/s |
| TensorRT | FP16 | 1 | **3.22 ms** | **3.45 ms** | **310.8 images/s** |
| TensorRT | FP16 | 4 | **10.31 ms** | **11.41 ms** | **388.0 images/s** |
| TensorRT | FP16 | 8 | **24.84 ms** | **29.93 ms** | **322.1 images/s** |

At batch 8, TensorRT FP16 provides approximately `2.74x` the PyTorch
throughput and `1.81x` the TensorRT FP32 throughput under this boundary.

## Memory Capacity

PyTorch and ONNX Runtime use the CUDA free-memory delta from immediately
before backend loading to after warmup. This includes runtime allocations and
allocator caching and is an observed capacity indicator, not an isolated
tensor-size calculation.

| Backend | Batch 1 | Batch 4 | Batch 8 |
| --- | ---: | ---: | ---: |
| PyTorch CUDA FP32 | 222 MiB | 572 MiB | 1,170 MiB |
| ONNX Runtime CUDA FP32 | 542 MiB | 1,042 MiB | 2,074 MiB |

TensorRT reports deterministic execution-context memory plus exact input and
output buffers. This is a lower bound that excludes engine weights, CUDA
context overhead, and allocator caching.

| TensorRT precision | Batch 1 | Batch 4 | Batch 8 |
| --- | ---: | ---: | ---: |
| FP32 | 531 MiB | 552 MiB | 580 MiB |
| FP16 | 269 MiB | 290 MiB | 318 MiB |

Memory values produced by the two methods should not be ranked as if they were
the same metric. Raw reports retain the method and byte counts.

## Numerical Acceptance

ONNX Runtime CPU logits are the TensorRT numerical reference. The deployment
probability threshold `0.80` is converted to the equivalent logit threshold
`1.386294`.

| Precision | Maximum absolute error | Mean absolute error | Mask mismatch rate | Result |
| --- | ---: | ---: | ---: | --- |
| FP32 | `3.81e-05` | `1.20e-06` | `0` | Pass |
| FP16 | `1.31` | `2.29e-02` | `1.34e-05` | Pass |

The larger FP16 logit error is acceptable because the declared gate is based
on both numerical error and thresholded segmentation behavior. The maximum
observed mismatch rate remains below `1e-04`.

## Portability

TensorRT plans are tied to the GPU architecture, TensorRT version, CUDA
runtime, platform, and selected tactics. The x86 RTX 4070 engines are not
deployed to Jetson. The ONNX graph is transferred and rebuilt on the Jetson
Orin Nano target.

## Reproduction

Build and benchmark TensorRT:

```bash
conda activate trt310
python scripts/build_tensorrt.py --precision both
python scripts/benchmark_tensorrt.py
```

Run the cross-backend tensor I/O benchmarks:

```bash
conda activate defect310
PYTHONPATH=src python scripts/benchmark_backend.py \
  --backend pytorch-cuda \
  --report outputs/reports/pytorch_cuda_benchmark.json

conda activate ortgpu310
PYTHONPATH=src python scripts/benchmark_backend.py \
  --backend onnxruntime-cuda \
  --report outputs/reports/onnxruntime_cuda_benchmark.json
```

Machine-readable artifacts:

- `outputs/reports/tensorrt_build.json`
- `outputs/reports/tensorrt_benchmark.json`
- `outputs/reports/pytorch_cuda_benchmark.json`
- `outputs/reports/onnxruntime_cuda_benchmark.json`
