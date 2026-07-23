# System Design

## Problem Statement

Steel surface defects occupy a small fraction of long, high-resolution strip
images. Missing a defect is costly, while excessive false alarms reduce line
throughput. The system must therefore preserve small-defect quality while
meeting a documented latency and throughput target.

## Functional Requirements

For a valid image or bounded image batch, the service returns:

- model and schema version;
- per-class confidence;
- run-length encoded masks;
- connected-component bounding boxes and defect areas;
- preprocessing, queue, inference, postprocessing, and request timing.

The service also provides liveness, readiness, model metadata, and metrics
endpoints.

## Non-Functional Requirements

- Reproducible data splits, configuration, and artifact hashes.
- Explicit payload, batch, timeout, and concurrency limits.
- Stable versioned API schemas.
- Numerical parity checks before an exported model is promoted.
- Graceful errors for invalid input, timeout, and model unavailability.
- Structured logs and metrics sufficient to explain latency regressions.
- CPU-only unit tests separated from GPU integration tests.

## Component Boundary

```text
Client
  -> FastAPI gateway
      -> request validation
      -> image decoding and normalization
      -> Triton HTTP/gRPC client
  -> Triton scheduler
      -> dynamic batching
      -> TensorRT backend
  -> postprocessing
      -> thresholding and connected components
      -> RLE and JSON response
```

Triton is the only component that owns GPU execution. The gateway does not load
a second copy of the model.

## Failure Policy

| Failure | Response |
| --- | --- |
| Unsupported media type | `415` with stable error code |
| Invalid or corrupt image | `422` with validation details |
| Payload or batch too large | `413` |
| Triton unavailable or model not ready | `503` |
| Inference deadline exceeded | `504` |
| Unexpected internal failure | `500`, correlation ID, no stack trace in response |

## Artifact Promotion

An artifact is eligible for the service only when:

1. the source checkpoint and dataset manifest are identified by hash;
2. ONNX export passes structural validation;
3. parity tests pass on the fixed regression set;
4. TensorRT engine build settings are recorded;
5. quality loss remains within the declared tolerance;
6. the model passes smoke, load, and failure-path tests.

## Acceptance Criteria

1. Raw data validation detects missing files, malformed RLE, and schema drift.
2. Training and evaluation run from versioned configuration without path edits.
3. PyTorch, ONNX Runtime CUDA, and TensorRT outputs pass parity tolerances.
4. FP32 and FP16 quality and performance are compared on the same test set.
5. Triton exposes a ready model over HTTP/gRPC with dynamic batching.
6. Gateway contract tests cover success and expected failure paths.
7. Load tests report QPS, P50/P95/P99, failures, and server-side timing.
8. A clean checkout can reproduce the documented CPU validation workflow.

## Risks

| Risk | Mitigation |
| --- | --- |
| Sparse masks and class imbalance | Dice+BCE loss, class-aware split, per-class recall |
| Long aspect ratio and small defects | Preserve strip geometry and compare tiling with resize |
| Competition-data terms | Do not redistribute data; document access and terms |
| Overfitting | Fixed test set, early stopping, versioned experiment reports |
| Unfair backend benchmark | Freeze preprocessing, input shape, warmup, runs, and precision |
| CUDA/TensorRT version mismatch | Record environment and keep GPU packages outside generic dependencies |
| Gateway duplicates GPU work | Keep model ownership in Triton and test the boundary |
