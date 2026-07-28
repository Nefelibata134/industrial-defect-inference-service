# Triton Serving Stack

## Runtime Compatibility

The TensorRT plan is loaded by a pinned Triton image. TensorRT plans are
compiled artifacts rather than portable model graphs, so the server runtime
must remain compatible with the TensorRT version, CUDA stack, GPU
architecture, and optimization profile used during the build.

| Component | Version |
| --- | --- |
| Triton Inference Server | 2.56.0 |
| Container | `nvcr.io/nvidia/tritonserver:25.03-py3` |
| TensorRT | 10.9.0.34 |
| CUDA | 12.8.1 |
| Engine input profile | batch 1 / 4 / 8, FP32, `3 x 256 x 1024` |
| Engine output | FP32, `4 x 256 x 1024` |

Changing the Triton image or target GPU requires rebuilding and revalidating
the engine. The ONNX graph remains the portable promotion boundary.

## Model Repository

```text
model_repository/
└── steel_defect_segmentation/
    ├── config.pbtxt
    └── 1/
        ├── artifact.json
        └── model.plan
```

The generated plan and artifact manifest are excluded from Git. Prepare the
repository from a locally validated engine:

```bash
python scripts/prepare_triton_repository.py \
  --engine models/unet_resnet18_severstal_fp16.plan
```

The command copies the plan into model version `1`, records its SHA256, byte
size, and source path, then verifies the repository contract.

## Dynamic Batching

The model configuration accepts batches from `1` through `8` and asks the
Triton scheduler to prefer batches `4` and `8`:

```protobuf
dynamic_batching {
  preferred_batch_size: [4, 8]
  max_queue_delay_microseconds: 2000
}
```

Concurrent single-image requests may wait for at most two milliseconds while
Triton attempts to combine them. This can increase throughput and GPU
utilization under load, but also adds queue latency at low traffic. The queue
delay is therefore a benchmark parameter, not an unconditional optimization.

## Service Boundary

```text
client
  -> FastAPI: media type, byte limit, decode, dimension validation
  -> preprocessing: resize, RGB conversion, normalization, NCHW batch
  -> Triton HTTP: scheduling and TensorRT GPU execution
  -> postprocessing: sigmoid, threshold, nearest-neighbor mask resize
  -> JSON response: RLE, connected components, class results, stage timing
```

Triton is the only component that loads the TensorRT plan. The gateway remains
stateless and can be scaled independently.

## Local Deployment

Prerequisites:

- Docker Engine with Compose support;
- NVIDIA Container Toolkit or Docker Desktop GPU support;
- an NVIDIA driver compatible with CUDA 12.8;
- a prepared model repository.

Start the stack:

```bash
docker compose up --build -d
docker compose ps
```

Validate the endpoints:

```bash
curl --fail http://localhost:8000/v2/health/ready
curl --fail http://localhost:8080/health/ready
curl --fail http://localhost:9090/-/ready

curl --fail \
  -F image=@data/raw/severstal/train_images/0002cc93b.jpg \
  "http://localhost:8080/v1/segment?threshold=0.80"

python scripts/smoke_service.py \
  --image data/raw/severstal/train_images/0002cc93b.jpg
```

Stop the stack:

```bash
docker compose down
```

## Observability

Prometheus scrapes both the Triton metrics endpoint and the FastAPI gateway.

| Target | Metrics |
| --- | --- |
| `triton:8002` | request count, queue time, compute stages, GPU utilization |
| `api:8080/metrics` | HTTP count/status, end-to-end latency, Triton call latency |

Useful gateway series:

- `gateway_http_requests_total`
- `gateway_http_request_duration_seconds`
- `gateway_model_inference_duration_seconds`

The gateway response also returns per-request preprocessing, inference,
postprocessing, and total time in milliseconds. Prometheus data is used for
aggregate service behavior; response timing is used for individual request
diagnosis.

## Failure Contract

| Condition | HTTP status |
| --- | ---: |
| Unsupported media type | 415 |
| Payload exceeds 8 MiB | 413 |
| Decode failure or dimensions other than `1600 x 256` | 422 |
| Triton unavailable, model unavailable, or inference failure | 503 |

CPU contract tests replace Triton with an asynchronous fake client. Container
smoke tests exercise the real TensorRT plan and GPU runtime.
