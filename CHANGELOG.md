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
- CPU unit tests, Ruff checks, and GitHub Actions workflow.

### Planned

- Frozen validation/test quality report for the promoted checkpoint.
- ONNX parity and TensorRT FP32/FP16 comparison.
- Triton model repository and FastAPI gateway.
- Docker Compose, observability, and load-test report.
