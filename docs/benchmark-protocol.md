# Benchmark Protocol

## Objective

Measure model quality and system performance without conflating model,
precision, batching, transfer, or service overhead.

## Backends

- PyTorch CUDA FP32.
- ONNX Runtime CUDA FP32.
- TensorRT FP32.
- TensorRT FP16.
- Triton TensorRT backend under controlled concurrency.

## Fixed Variables

- Dataset manifest and model checkpoint.
- Image decoding, normalization, resizing, and output thresholds.
- Input tensor shape and batch size for like-for-like comparisons.
- Hardware, driver, CUDA, TensorRT, clock, and power state.
- Warmup and measured request counts.
- Synchronization points and timing boundary.

## Quality Gates

Exported backends are evaluated against the same regression and test samples.
Reports include:

- maximum and mean absolute tensor error against PyTorch;
- per-class Dice and IoU;
- per-class precision and recall;
- image-level false-negative rate;
- threshold values selected only from the validation set.

The acceptable FP16 quality delta must be declared before the final test run.

## Latency Boundaries

| Name | Included work |
| --- | --- |
| Model latency | Synchronized device inference only |
| Pipeline latency | Decode, preprocess, transfer, inference, postprocess |
| Service latency | HTTP/gRPC serialization, queue, pipeline, response |

Each report states the exact boundary. FPS is never derived from an
underspecified timer.

## Required Report Fields

- Git commit, config hash, dataset-manifest hash, and model SHA256.
- GPU, driver, CUDA, cuDNN, TensorRT, Python, and package versions.
- Backend, precision, input shape, batch, warmup, and measured runs.
- Mean, minimum, maximum, P50, P95, and P99 latency.
- Throughput, peak GPU memory, and GPU utilization.
- Included and excluded preprocessing, transfer, queue, and postprocessing.

When a backend cannot expose allocator-level peak memory through a common
API, the report states the exact alternative method. CUDA free-memory deltas
and TensorRT execution-context plus I/O capacity are retained as different
metrics and are not ranked as if they were equivalent.

## Initial Model Benchmark

```text
input shape: N x 3 x 256 x 1024
batch sizes: 1, 4, 8
warmup runs: 50
measured runs: 500
```

## Service Load Test

Triton and gateway tests additionally report:

- protocol and payload format;
- client and server host placement;
- concurrency levels 1, 2, 4, 8, and 16;
- request count and duration;
- QPS and P50/P95/P99 response latency;
- success, client-error, timeout, and server-error counts;
- Triton queue, compute-input, compute-inference, and compute-output time;
- observed dynamic batch sizes.

The final report includes raw machine-readable output and a concise comparison
table. Results are not copied between different hardware or engine builds.

The initial TensorRT and cross-backend result is documented in
[`tensorrt-benchmark.md`](tensorrt-benchmark.md).
