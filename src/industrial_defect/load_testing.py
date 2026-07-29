from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

TRITON_COUNTER_NAMES = (
    "nv_inference_request_success",
    "nv_inference_count",
    "nv_inference_exec_count",
    "nv_inference_request_duration_us",
    "nv_inference_queue_duration_us",
    "nv_inference_compute_input_duration_us",
    "nv_inference_compute_infer_duration_us",
    "nv_inference_compute_output_duration_us",
)


@dataclass(frozen=True)
class ServiceRequestResult:
    status_code: int | None
    latency_ms: float
    server_timing_ms: dict[str, float] | None = None
    error: str | None = None

    @property
    def successful(self) -> bool:
        return self.error is None and self.status_code is not None and 200 <= self.status_code < 300


def _latency_distribution(values_ms: Sequence[float]) -> dict[str, float]:
    if not values_ms:
        raise ValueError("latency values must be non-empty")

    values = np.asarray(values_ms, dtype=np.float64)
    return {
        "mean_ms": float(values.mean()),
        "min_ms": float(values.min()),
        "max_ms": float(values.max()),
        "p50_ms": float(np.percentile(values, 50)),
        "p95_ms": float(np.percentile(values, 95)),
        "p99_ms": float(np.percentile(values, 99)),
    }


def _format_optional_metric(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3f}"


def summarize_service_run(
    results: Sequence[ServiceRequestResult],
    *,
    elapsed_seconds: float,
    concurrency: int,
) -> dict[str, Any]:
    if not results:
        raise ValueError("request results must be non-empty")
    if elapsed_seconds <= 0:
        raise ValueError("elapsed_seconds must be positive")
    if concurrency <= 0:
        raise ValueError("concurrency must be positive")

    successful = [result for result in results if result.successful]
    failed = len(results) - len(successful)
    status_counts = Counter(
        str(result.status_code) if result.status_code is not None else "transport_error"
        for result in results
    )

    latency = (
        _latency_distribution([result.latency_ms for result in successful]) if successful else None
    )

    timing_keys = sorted(
        {
            key
            for result in successful
            if result.server_timing_ms is not None
            for key in result.server_timing_ms
        }
    )
    server_timing_mean_ms = {
        key: float(
            np.mean(
                [
                    result.server_timing_ms[key]
                    for result in successful
                    if result.server_timing_ms is not None and key in result.server_timing_ms
                ]
            )
        )
        for key in timing_keys
    }

    return {
        "concurrency": concurrency,
        "requests": len(results),
        "successful_requests": len(successful),
        "failed_requests": failed,
        "error_rate": failed / len(results),
        "elapsed_seconds": elapsed_seconds,
        "successful_qps": len(successful) / elapsed_seconds,
        "status_counts": dict(sorted(status_counts.items())),
        "client_latency": latency,
        "server_timing_mean_ms": server_timing_mean_ms,
    }


def parse_triton_counters(
    metrics_text: str,
    *,
    model_name: str,
    model_version: str,
) -> dict[str, float]:
    counters: dict[str, float] = {}
    expected_labels = f'model="{model_name}",version="{model_version}"'

    for line in metrics_text.splitlines():
        if line.startswith("#") or "{" not in line:
            continue
        metric_and_labels, separator, raw_value = line.rpartition(" ")
        if not separator:
            continue
        metric_name, _, labels = metric_and_labels.partition("{")
        if metric_name in TRITON_COUNTER_NAMES and labels.rstrip("}") == expected_labels:
            counters[metric_name] = float(raw_value)

    missing = sorted(set(TRITON_COUNTER_NAMES) - counters.keys())
    if missing:
        raise ValueError(f"missing Triton metrics: {', '.join(missing)}")
    return counters


def summarize_triton_counters(
    before: dict[str, float],
    after: dict[str, float],
) -> dict[str, float | None]:
    deltas = {name: after[name] - before[name] for name in TRITON_COUNTER_NAMES}
    if any(value < 0 for value in deltas.values()):
        raise ValueError("Triton counters decreased during the benchmark")

    request_count = deltas["nv_inference_request_success"]
    inference_count = deltas["nv_inference_count"]
    execution_count = deltas["nv_inference_exec_count"]
    if request_count <= 0 or inference_count <= 0 or execution_count <= 0:
        raise ValueError("Triton did not record successful model executions")

    per_request = 1000.0 * request_count
    request_ms = deltas["nv_inference_request_duration_us"] / per_request

    def validated_duration(name: str) -> float | None:
        duration_ms = deltas[name] / per_request
        if duration_ms < 0 or duration_ms > request_ms:
            return None
        return duration_ms

    input_ms = validated_duration("nv_inference_compute_input_duration_us")
    infer_ms = validated_duration("nv_inference_compute_infer_duration_us")
    output_ms = validated_duration("nv_inference_compute_output_duration_us")
    compute_parts = (input_ms, infer_ms, output_ms)
    compute_total_ms = (
        sum(part for part in compute_parts if part is not None)
        if all(part is not None for part in compute_parts)
        else None
    )

    return {
        "successful_requests": request_count,
        "inference_count": inference_count,
        "execution_count": execution_count,
        "average_batch_size": inference_count / execution_count,
        "queue_ms_per_request": (deltas["nv_inference_queue_duration_us"] / per_request),
        "request_ms_per_request": request_ms,
        "compute_input_ms_per_request": input_ms,
        "compute_infer_ms_per_request": infer_ms,
        "compute_output_ms_per_request": output_ms,
        "compute_total_ms_per_request": compute_total_ms,
    }


def render_service_benchmark_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Service Load Test",
        "",
        f"- Endpoint: `{report['endpoint']}`",
        f"- Image: `{report['image']}`",
        f"- Threshold: `{report['threshold']}`",
        f"- Warmup requests per level: `{report['warmup_requests_per_level']}`",
        f"- Measured requests per level: `{report['requests_per_level']}`",
        "",
        "| Concurrency | Success | Errors | Error rate | QPS | P50 ms | P95 ms | P99 ms |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    for result in report["results"]:
        latency = result["client_latency"]
        if latency is None:
            p50 = p95 = p99 = "n/a"
        else:
            p50 = f"{latency['p50_ms']:.2f}"
            p95 = f"{latency['p95_ms']:.2f}"
            p99 = f"{latency['p99_ms']:.2f}"
        lines.append(
            "| "
            f"{result['concurrency']} | "
            f"{result['successful_requests']} | "
            f"{result['failed_requests']} | "
            f"{result['error_rate']:.2%} | "
            f"{result['successful_qps']:.2f} | "
            f"{p50} | {p95} | {p99} |"
        )

    if all("triton_metrics" in result for result in report["results"]):
        lines.extend(
            (
                "",
                "| Concurrency | Triton requests | Executions | Avg batch | "
                "Queue ms | Request ms | GPU infer ms | Output ms |",
                "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            )
        )
        for result in report["results"]:
            metrics = result["triton_metrics"]
            lines.append(
                "| "
                f"{result['concurrency']} | "
                f"{metrics['successful_requests']:.0f} | "
                f"{metrics['execution_count']:.0f} | "
                f"{metrics['average_batch_size']:.2f} | "
                f"{metrics['queue_ms_per_request']:.3f} | "
                f"{metrics['request_ms_per_request']:.3f} | "
                f"{_format_optional_metric(metrics['compute_infer_ms_per_request'])} | "
                f"{_format_optional_metric(metrics['compute_output_ms_per_request'])} |"
            )

    lines.extend(
        (
            "",
            "Client latency includes request serialization, transport, gateway processing, "
            "Triton inference, postprocessing, and response transfer.",
            "Triton timings come from server counters and isolate queueing and GPU "
            "execution from gateway transport overhead.",
            "",
        )
    )
    return "\n".join(lines)
