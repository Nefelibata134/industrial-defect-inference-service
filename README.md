# Industrial Defect Inference Service

Production-style GPU inference pipeline for pixel-level steel surface defect
segmentation.

The repository covers the complete path from raw data validation to a
containerized, observable inference service:

```text
Severstal steel images and RLE masks
  -> deterministic train/validation/test manifests
  -> PyTorch segmentation model
  -> ONNX export and numerical parity checks
  -> TensorRT FP32/FP16 engines
  -> NVIDIA Triton Inference Server
  -> FastAPI gateway
  -> load tests and Prometheus metrics
```

## Engineering Scope

- Validate source files, RLE masks, image dimensions, and class distribution.
- Build reproducible multilabel-aware data splits with file hashes.
- Train a lightweight U-Net with a ResNet-18 encoder.
- Report per-class Dice, IoU, recall, image-level false-negative rate, and
  calibration thresholds.
- Verify PyTorch-to-ONNX output parity on a fixed regression set.
- Build and benchmark TensorRT FP32 and FP16 engines.
- Serve TensorRT through Triton with HTTP/gRPC, health checks, and dynamic
  batching.
- Expose a versioned FastAPI contract for image validation and mask encoding.
- Compare model latency, end-to-end latency, throughput, GPU memory, and error
  rate under controlled concurrency.
- Package the gateway and inference server with Docker Compose.
- Run linting and CPU unit tests in GitHub Actions.

## Dataset

The primary dataset is
[Severstal: Steel Defect Detection](https://www.kaggle.com/c/severstal-steel-defect-detection/data).
Its labeled training set contains 12,568 grayscale steel-strip images at
1600 x 256 pixels. Four anonymized defect classes are represented as
run-length encoded pixel masks, with substantial class and foreground
imbalance.

Dataset files are not redistributed. A local copy is obtained under the
Kaggle competition terms and validated with the repository tooling. See
[`docs/dataset-card.md`](docs/dataset-card.md) and
[`docs/adr/0001-dataset-and-task.md`](docs/adr/0001-dataset-and-task.md).

Validated source statistics:

| Item | Count |
| --- | ---: |
| Training images | 12,568 |
| Images with at least one defect | 6,666 |
| Normal images | 5,902 |
| Positive masks | 7,095 |
| Class 1 / 2 / 3 / 4 masks | 897 / 247 / 5,150 / 801 |

## Architecture

```text
                     +----------------------+
image request ------>| FastAPI gateway      |
                     | validation / encoding |
                     +----------+-----------+
                                |
                                | Triton HTTP/gRPC
                                v
                     +----------------------+
                     | Triton Inference     |
                     | TensorRT backend     |
                     | dynamic batching     |
                     +----------+-----------+
                                |
              +-----------------+-----------------+
              |                                   |
              v                                   v
      segmentation mask                  Prometheus metrics
      class confidence                   latency / queue / GPU
```

Triton owns model scheduling and GPU execution. The FastAPI gateway owns the
public request schema, image validation, preprocessing coordination, mask
serialization, and application-level errors. The boundary is documented in
[`docs/system-design.md`](docs/system-design.md).

## API Contract

Planned stable endpoints:

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/health/live` | Process liveness |
| `GET` | `/health/ready` | Gateway and model readiness |
| `POST` | `/v1/segment` | Segment defects in one image |
| `POST` | `/v1/segment:batch` | Segment a bounded image batch |

The response includes model version, per-class confidence, encoded masks,
defect area, and request/inference timing. Invalid media types, oversized
payloads, unavailable models, and timeouts return explicit error codes.

## Bootstrap

The reference development environment is Ubuntu 22.04 under WSL2 with Python
3.10 and an NVIDIA RTX 4070 Laptop GPU.

```bash
git clone https://github.com/Nefelibata134/industrial-defect-inference-service.git
cd industrial-defect-inference-service

conda create -n defect310 python=3.10 -y
conda activate defect310
python -m pip install --upgrade pip
python -m pip install -e ".[data,train,dev]"

ruff check .
pytest -q
python scripts/check_environment.py
```

GPU framework packages are installed separately so their CUDA and TensorRT
versions match the host runtime. After accepting the competition rules,
configure the Kaggle CLI credentials in its standard user configuration path
and run:

```bash
python scripts/download_dataset.py
```

Place the Kaggle files locally:

```text
data/raw/severstal/
├── train.csv
└── train_images/
    ├── 0002cc93b.jpg
    └── ...
```

Validate the raw dataset before generating split manifests:

```bash
python scripts/inspect_dataset.py \
  --config configs/project.yaml \
  --data-root data/raw/severstal \
  --output outputs/reports/dataset_report.json

python scripts/build_manifest.py \
  --output data/manifests/severstal_v1.csv \
  --summary outputs/reports/split_summary.json

python scripts/visualize_masks.py --count 4
```

The split is deterministic with seed `42` and contains 8,798 training, 1,885
validation, and 1,885 test images. Aggregate validation and split reports are
versioned; raw data and rendered source-image samples remain local.

## Training Baseline

The baseline is a four-channel U-Net with an ImageNet-pretrained ResNet-18
encoder. It uses BCE-with-logits plus soft Dice loss, AdamW, automatic mixed
precision, validation macro Dice checkpoint selection, learning-rate
reduction, and early stopping.

Inspect the input contract and run a single optimization step before starting
a full experiment:

```bash
python scripts/inspect_training_batch.py --batch-size 4
python scripts/smoke_train_step.py --device cuda --batch-size 2
```

Start the configured baseline and render its curves:

```bash
python scripts/train.py --device cuda
python scripts/plot_training_history.py
```

Sweep validation thresholds before changing the model or loss:

```bash
python scripts/evaluate_thresholds.py \
  --device cuda \
  --checkpoint models/best_unet_resnet18.pt
```

Run the class-aware sampling experiment while keeping the baseline loss and
model unchanged:

```bash
python scripts/train.py \
  --device cuda \
  --epochs 5 \
  --sampler class-aware \
  --sampling-power 0.5 \
  --checkpoint models/class_aware_e05_best_unet_resnet18.pt \
  --latest-checkpoint models/class_aware_e05_latest_unet_resnet18.pt \
  --history outputs/reports/class_aware_e05_training_history.json
```

Run the focal-loss experiment while retaining the class-aware sampler:

```bash
python scripts/train.py \
  --device cuda \
  --epochs 5 \
  --sampler class-aware \
  --sampling-power 0.5 \
  --loss focal-dice \
  --focal-alpha 0.75 \
  --focal-gamma 2.0 \
  --checkpoint models/focal_class_aware_e05_best_unet_resnet18.pt \
  --latest-checkpoint models/focal_class_aware_e05_latest_unet_resnet18.pt \
  --history outputs/reports/focal_class_aware_e05_training_history.json
```

Resume an interrupted run from the latest completed epoch:

```bash
python scripts/train.py \
  --device cuda \
  --resume models/latest_unet_resnet18.pt
```

The best checkpoint for model selection and the latest checkpoint for
interruption recovery are written under `models/`. Structured epoch history
and rendered curves are written under `outputs/reports/`. These generated
artifacts are excluded from source control and summarized in a
versioned quality report after the experiment is reproduced.

## Repository Layout

```text
configs/                 Versioned experiment and service configuration
data/                    Local datasets and generated manifests
docs/                    Design, dataset, benchmark, and decision records
models/                  Local checkpoints, ONNX files, and TensorRT engines
outputs/                 Reports, visualizations, and load-test results
scripts/                 Reproducible command-line entry points
src/industrial_defect/   Reusable data, model, runtime, and service code
tests/                   Unit and contract tests
```

Large datasets, checkpoints, and generated engines are excluded from Git.
Release artifacts are identified by model version, Git commit, and SHA256.

## Evaluation

Model quality:

- Per-class and macro Dice.
- Per-class and macro IoU.
- Per-class recall and precision.
- Image-level false-negative rate.
- Threshold-selection report on the validation set.

Deployment:

- Model-only and end-to-end mean, P50, P95, and P99 latency.
- Throughput at documented batch and concurrency settings.
- Triton queue, compute-input, compute-inference, and compute-output time.
- GPU utilization and peak memory.
- Request success/error rate and serialized model size.

Backend comparisons use identical inputs, preprocessing, postprocessing,
thresholds, warmup, hardware state, and precision. The full protocol is in
[`docs/benchmark-protocol.md`](docs/benchmark-protocol.md).

## Release Milestones

| Version | Deliverable |
| --- | --- |
| `v0.1` | Data contract, validation report, deterministic split manifests |
| `v0.2` | Reproducible PyTorch baseline and quality report |
| `v0.3` | ONNX parity checks and TensorRT FP32/FP16 benchmarks |
| `v0.4` | Triton model repository, FastAPI contract, and Docker Compose |
| `v1.0` | Load-test report, observability evidence, demo, and release artifacts |

Progress is represented by versioned code, tests, benchmark artifacts, release
notes, and Git history. Results are added only after they are reproduced.

## Licensing

This repository does not redistribute the Severstal dataset, third-party model
weights, CUDA, TensorRT, or Triton binaries. Dataset access remains subject to
the Kaggle competition terms. Third-party dependency licenses apply
independently. A project code license will be selected before the first public
release.
