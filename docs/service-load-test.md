# Service Load Test

## Scope

This report measures the complete HTTP request boundary:

```text
multipart upload
  -> FastAPI validation and preprocessing
  -> Triton HTTP request
  -> TensorRT FP16 execution
  -> sigmoid, thresholding, connected components, and RLE
  -> JSON response transfer
```

It is a service-capacity result, not a model-only latency result. The client
and containers ran on the same WSL2 host. Benchmark clients ignored proxy
environment variables.

## Environment

| Component | Value |
| --- | --- |
| GPU | NVIDIA GeForce RTX 4070 Laptop GPU, 8188 MiB |
| Driver | 610.47 |
| Host | WSL2, Ubuntu 22.04 |
| Python | 3.10.20 |
| Triton | 2.56.0, `nvcr.io/nvidia/tritonserver:25.03-py3` |
| TensorRT | 10.9.0.34 |
| Engine | FP16, dynamic batch profile 1 / 4 / 8 |
| Input | FP32, `3 x 256 x 1024` |
| Output | FP32 logits, `4 x 256 x 1024` |
| Threshold | 0.80 |

The output contains 1,048,576 FP32 values, approximately 4 MiB per image
before HTTP serialization.

## Method

The test used one Severstal image at concurrency levels 1, 2, 4, and 8.
Each level received five warmup requests followed by 100 measured requests.
The load client used a separate HTTP connection per worker and reported
successful QPS plus client-side P50, P95, and P99 latency.

```bash
python scripts/load_test_service.py \
  --image data/raw/severstal/train_images/0002cc93b.jpg \
  --concurrency 1 2 4 8 \
  --warmup 5 \
  --requests 100 \
  --report outputs/reports/service_load_test.json \
  --markdown outputs/reports/service_load_test.md
```

## Results

All 400 measured requests returned HTTP 200.

| Concurrency | Success | Errors | QPS | P50 ms | P95 ms | P99 ms |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 100 | 0 | 3.16 | 314.22 | 338.93 | 349.25 |
| 2 | 100 | 0 | 3.56 | 553.04 | 594.27 | 603.99 |
| 4 | 100 | 0 | 3.64 | 1080.18 | 1194.73 | 1220.48 |
| 8 | 100 | 0 | 3.79 | 2095.35 | 2154.96 | 2198.43 |

| Concurrency | Triton requests | Executions | Observed average batch | Queue ms/request |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 100 | 100 | 1.00 | 2.148 |
| 2 | 100 | 100 | 1.00 | 2.155 |
| 4 | 100 | 70 | 1.43 | 337.755 |
| 8 | 100 | 33 | 3.03 | 707.671 |

Triton compute-output counters became internally inconsistent at concurrency
levels above one. The load client rejects impossible counter deltas and
records them as unavailable instead of reporting fabricated stage timings.
Capacity conclusions therefore use client latency, successful QPS, request
counts, execution counts, and valid queue metrics.

## Interpretation

Throughput increased from 3.16 to 3.79 QPS between concurrency 1 and 8, a
19.9% gain. Over the same range, P95 latency increased from 338.93 ms to
2154.96 ms. The service reaches its practical saturation region near 3.8 QPS;
additional concurrency primarily increases queueing and response latency.

Dynamic batching was also compared against a configuration with batching
disabled at concurrency 4:

| Configuration | QPS | P95 ms |
| --- | ---: | ---: |
| Dynamic batching, preferred 4 / 8 | 3.56 | 1144.13 |
| Dynamic batching disabled | 3.43 | 1199.69 |

Dynamic batching provided a small throughput and tail-latency improvement, so
the 2 ms batching policy remains enabled. The small gain indicates that the
dense-output HTTP boundary, rather than TensorRT engine execution, is the
dominant scaling limit.

## Capacity Recommendation

- Use concurrency 1 for latency-sensitive operation.
- Use concurrency 2 when a modest throughput increase is worth approximately
  0.6 seconds of P95 latency.
- Treat concurrency 4 and 8 as overload behavior for a single service
  instance with the current response contract.
- Scale out gateway and Triton instances only after profiling GPU memory and
  contention on the target host.

Potential follow-up optimizations are Triton shared memory, gRPC transport,
server-side thresholding and encoding, or a backend that returns compact masks
instead of dense FP32 logits. These changes require a new parity and capacity
test because they alter the service boundary.

The failure and restart behavior is documented in
[`serving.md`](serving.md#recovery-verification).
