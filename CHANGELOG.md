# Changelog

All notable changes to this repository are documented in this file.

## [1.0.0] - 2026-07-29

### Added

- Deterministic Severstal dataset validation and multilabel-aware manifests.
- U-Net ResNet-18 training pipeline with mixed precision, checkpoint recovery,
  class-aware sampling, and threshold analysis.
- Per-class Dice, IoU, precision, recall, and image-level failure reports.
- ONNX export with PyTorch and ONNX Runtime numerical parity checks.
- TensorRT FP32 and FP16 engine builds, optimization profiles, runtime
  validation, and backend benchmarks.
- NVIDIA Triton model repository with dynamic batching.
- FastAPI inference gateway with validation, mask encoding, Prometheus
  metrics, and Docker Compose deployment.
- Concurrent service load test with machine-readable results and capacity
  recommendations.
- Model-call deadlines and verified 503/recovery behavior when Triton is
  unavailable.
- Unit, contract, deployment, and parity checks in the continuous integration
  workflow.
- MIT License for project source code and documentation.

## Pre-1.0.0 Development Record

### Added

- Raw Severstal CSV and image validation contract.
- Kaggle download/extraction command with archive integrity and path checks.
- RLE encode/decode utilities with round-trip and malformed-input tests.
- Deterministic label-powerset train/validation/test split generation.
- Dataset aggregate report, split manifest, and split summary hashes.
- Local mask-overlay validation command.
- Versioned project configuration.
- System design, dataset card, benchmark protocol, and architecture decisions.
- U-Net ResNet-18 baseline with AMP, checkpoint recovery, and early stopping.
- Threshold sweeps, per-image error analysis, and failure visualization.
- Validation ablation report and promoted-checkpoint selection policy.
- Reproducible checkpoint-to-ONNX export with embedded artifact metadata.
- ONNX graph inspection, structural validation, and dynamic-batch contract
  tests.
- PyTorch-to-ONNX Runtime logit, mask, and metric parity report.
- TensorRT FP32/FP16 engine build, numerical validation, and backend benchmark.
- Versioned Triton model repository with artifact integrity metadata.
- Dynamic batching for batches up to eight with preferred sizes four and eight.
- FastAPI image validation, Triton HTTP inference, RLE response serialization,
  health checks, and Prometheus metrics.
- Docker Compose stack for Triton, the API gateway, and Prometheus.
- Gateway, postprocessing, and model-repository contract tests.
- CPU unit tests, Ruff checks, and GitHub Actions workflow.

### Completed for 1.0.0

- Frozen validation/test quality report for the promoted checkpoint.
- Concurrent load-test report and final observability evidence.
