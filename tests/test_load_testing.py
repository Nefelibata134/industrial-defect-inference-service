import pytest

from industrial_defect.load_testing import (
    TRITON_COUNTER_NAMES,
    ServiceRequestResult,
    parse_triton_counters,
    render_service_benchmark_markdown,
    summarize_service_run,
    summarize_triton_counters,
)


def test_summarize_service_run_reports_qps_errors_and_tail_latency() -> None:
    results = [
        ServiceRequestResult(
            status_code=200,
            latency_ms=10.0,
            server_timing_ms={"inference": 4.0},
        ),
        ServiceRequestResult(
            status_code=200,
            latency_ms=20.0,
            server_timing_ms={"inference": 6.0},
        ),
        ServiceRequestResult(status_code=503, latency_ms=30.0, error="unavailable"),
    ]

    summary = summarize_service_run(
        results,
        elapsed_seconds=0.5,
        concurrency=2,
    )

    assert summary["successful_requests"] == 2
    assert summary["failed_requests"] == 1
    assert summary["error_rate"] == pytest.approx(1 / 3)
    assert summary["successful_qps"] == 4.0
    assert summary["status_counts"] == {"200": 2, "503": 1}
    assert summary["client_latency"]["p50_ms"] == 15.0
    assert summary["server_timing_mean_ms"]["inference"] == 5.0


def test_summarize_service_run_handles_transport_errors() -> None:
    summary = summarize_service_run(
        [
            ServiceRequestResult(
                status_code=None,
                latency_ms=5.0,
                error="connection refused",
            )
        ],
        elapsed_seconds=0.1,
        concurrency=1,
    )

    assert summary["successful_requests"] == 0
    assert summary["client_latency"] is None
    assert summary["status_counts"] == {"transport_error": 1}


@pytest.mark.parametrize(
    ("results", "elapsed_seconds", "concurrency", "message"),
    [
        ([], 1.0, 1, "non-empty"),
        ([ServiceRequestResult(200, 1.0)], 0.0, 1, "positive"),
        ([ServiceRequestResult(200, 1.0)], 1.0, 0, "positive"),
    ],
)
def test_summarize_service_run_rejects_invalid_inputs(
    results: list[ServiceRequestResult],
    elapsed_seconds: float,
    concurrency: int,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        summarize_service_run(
            results,
            elapsed_seconds=elapsed_seconds,
            concurrency=concurrency,
        )


def test_render_service_benchmark_markdown_contains_results() -> None:
    report = {
        "endpoint": "http://localhost:8080/v1/segment",
        "image": "sample.jpg",
        "threshold": 0.8,
        "warmup_requests_per_level": 5,
        "requests_per_level": 20,
        "results": [
            {
                "concurrency": 1,
                "successful_requests": 20,
                "failed_requests": 0,
                "error_rate": 0.0,
                "successful_qps": 10.0,
                "client_latency": {
                    "p50_ms": 50.0,
                    "p95_ms": 60.0,
                    "p99_ms": 70.0,
                },
                "triton_metrics": {
                    "successful_requests": 20.0,
                    "execution_count": 10.0,
                    "average_batch_size": 2.0,
                    "queue_ms_per_request": 1.0,
                    "request_ms_per_request": 8.0,
                    "compute_infer_ms_per_request": 3.0,
                    "compute_output_ms_per_request": 1.0,
                },
            }
        ],
    }

    markdown = render_service_benchmark_markdown(report)

    assert "| 1 | 20 | 0 | 0.00% | 10.00 | 50.00 | 60.00 | 70.00 |" in markdown
    assert "| 1 | 20 | 10 | 2.00 | 1.000 | 8.000 | 3.000 | 1.000 |" in markdown


def test_parse_and_summarize_triton_counters() -> None:
    labels = 'model="steel_defect_segmentation",version="1"'

    def metric_line(name: str, value: int) -> str:
        return f"{name}{{{labels}}} {value}"

    before_text = "\n".join(
        (
            metric_line("nv_inference_request_success", 10),
            metric_line("nv_inference_count", 10),
            metric_line("nv_inference_exec_count", 10),
            metric_line("nv_inference_request_duration_us", 10000),
            metric_line("nv_inference_queue_duration_us", 1000),
            metric_line("nv_inference_compute_input_duration_us", 2000),
            metric_line("nv_inference_compute_infer_duration_us", 3000),
            metric_line("nv_inference_compute_output_duration_us", 4000),
        )
    )
    after_text = "\n".join(
        (
            metric_line("nv_inference_request_success", 18),
            metric_line("nv_inference_count", 18),
            metric_line("nv_inference_exec_count", 12),
            metric_line("nv_inference_request_duration_us", 50000),
            metric_line("nv_inference_queue_duration_us", 9000),
            metric_line("nv_inference_compute_input_duration_us", 4000),
            metric_line("nv_inference_compute_infer_duration_us", 9000),
            metric_line("nv_inference_compute_output_duration_us", 8000),
        )
    )

    before = parse_triton_counters(
        before_text,
        model_name="steel_defect_segmentation",
        model_version="1",
    )
    after = parse_triton_counters(
        after_text,
        model_name="steel_defect_segmentation",
        model_version="1",
    )
    summary = summarize_triton_counters(before, after)

    assert summary["successful_requests"] == 8
    assert summary["execution_count"] == 2
    assert summary["average_batch_size"] == 4
    assert summary["queue_ms_per_request"] == 1
    assert summary["request_ms_per_request"] == 5
    assert summary["compute_infer_ms_per_request"] == 0.75
    assert summary["compute_total_ms_per_request"] == 1.5


def test_summarize_triton_counters_rejects_impossible_duration() -> None:
    before = {name: 0.0 for name in TRITON_COUNTER_NAMES}
    after = {
        "nv_inference_request_success": 1.0,
        "nv_inference_count": 1.0,
        "nv_inference_exec_count": 1.0,
        "nv_inference_request_duration_us": 10_000.0,
        "nv_inference_queue_duration_us": 1_000.0,
        "nv_inference_compute_input_duration_us": 1_000.0,
        "nv_inference_compute_infer_duration_us": 2_000.0,
        "nv_inference_compute_output_duration_us": 1e18,
    }

    summary = summarize_triton_counters(before, after)

    assert summary["request_ms_per_request"] == 10
    assert summary["compute_output_ms_per_request"] is None
    assert summary["compute_total_ms_per_request"] is None
