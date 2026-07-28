# Changelog

All notable engineering changes are documented here. Performance claims are
linked to reproducible reports and immutable model artifacts.

## Unreleased

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

### Planned

- Frozen validation/test quality report for the promoted checkpoint.
- Concurrent load-test report and final observability evidence.
