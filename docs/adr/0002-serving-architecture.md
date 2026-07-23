# ADR 0002: Separate the API Gateway from GPU Model Serving

- Status: Accepted
- Date: 2026-07-23

## Context

A single FastAPI process can load a TensorRT engine, but this couples request
validation, GPU lifecycle, scheduling, metrics, and scaling. It also makes
dynamic batching and backend-level timing harder to demonstrate.

## Decision

Use NVIDIA Triton Inference Server with the TensorRT backend as the model
server. Use FastAPI as a thin versioned gateway that owns validation,
application schemas, error mapping, and mask serialization.

Package the components with Docker Compose. Expose Triton health and Prometheus
metrics, and verify the service with both Triton Performance Analyzer and an
end-to-end HTTP load test.

## Consequences

- Model scheduling, concurrent execution, and dynamic batching are delegated
  to a purpose-built inference server.
- Gateway and model-server latency can be measured separately.
- The deployment has more moving parts and requires contract tests at the
  service boundary.
- Kubernetes is intentionally excluded until there is a real cluster,
  deployment topology, and operational requirement.
