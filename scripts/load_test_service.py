from __future__ import annotations

import argparse
import json
import mimetypes
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx

from industrial_defect.artifacts import sha256_file
from industrial_defect.benchmarking import system_metadata
from industrial_defect.load_testing import (
    ServiceRequestResult,
    parse_triton_counters,
    render_service_benchmark_markdown,
    summarize_service_run,
    summarize_triton_counters,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark the live segmentation gateway with concurrent image requests."
    )
    parser.add_argument("--base-url", default="http://localhost:8080")
    parser.add_argument(
        "--triton-metrics-url",
        default="http://localhost:8002/metrics",
    )
    parser.add_argument("--model-name", default="steel_defect_segmentation")
    parser.add_argument("--model-version", default="1")
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.80)
    parser.add_argument(
        "--concurrency",
        type=int,
        nargs="+",
        default=[1, 2, 4, 8],
    )
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--requests", type=int, default=100)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("outputs/reports/service_load_test.json"),
    )
    parser.add_argument(
        "--markdown",
        type=Path,
        default=Path("outputs/reports/service_load_test.md"),
    )
    return parser.parse_args()


def write_text_atomic(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def request_once(
    client: httpx.Client,
    *,
    endpoint: str,
    image_name: str,
    image_bytes: bytes,
    media_type: str,
    threshold: float,
) -> ServiceRequestResult:
    started = time.perf_counter()
    try:
        response = client.post(
            endpoint,
            params={"threshold": threshold},
            files={"image": (image_name, image_bytes, media_type)},
        )
        latency_ms = (time.perf_counter() - started) * 1000.0
        server_timing: dict[str, float] | None = None
        error: str | None = None
        if 200 <= response.status_code < 300:
            payload = response.json()
            raw_timing = payload.get("timing_ms")
            if isinstance(raw_timing, dict):
                server_timing = {
                    str(key): float(value)
                    for key, value in raw_timing.items()
                    if isinstance(value, int | float)
                }
        else:
            error = response.text[:500]
        return ServiceRequestResult(
            status_code=response.status_code,
            latency_ms=latency_ms,
            server_timing_ms=server_timing,
            error=error,
        )
    except (httpx.HTTPError, ValueError) as error:
        return ServiceRequestResult(
            status_code=None,
            latency_ms=(time.perf_counter() - started) * 1000.0,
            error=str(error),
        )


def read_triton_counters(
    *,
    metrics_url: str,
    model_name: str,
    model_version: str,
    timeout: float,
) -> dict[str, float]:
    with httpx.Client(timeout=timeout, trust_env=False) as client:
        response = client.get(metrics_url)
        response.raise_for_status()
    return parse_triton_counters(
        response.text,
        model_name=model_name,
        model_version=model_version,
    )


def run_level(
    *,
    endpoint: str,
    image_name: str,
    image_bytes: bytes,
    media_type: str,
    threshold: float,
    concurrency: int,
    warmup: int,
    requests: int,
    timeout: float,
    triton_metrics_url: str,
    model_name: str,
    model_version: str,
) -> dict[str, Any]:
    limits = httpx.Limits(
        max_connections=concurrency,
        max_keepalive_connections=concurrency,
    )
    with httpx.Client(timeout=timeout, limits=limits, trust_env=False) as client:
        request_arguments = {
            "endpoint": endpoint,
            "image_name": image_name,
            "image_bytes": image_bytes,
            "media_type": media_type,
            "threshold": threshold,
        }
        for _ in range(warmup):
            result = request_once(client, **request_arguments)
            if not result.successful:
                raise RuntimeError(
                    f"warmup failed: status={result.status_code}, error={result.error}"
                )

        triton_before = read_triton_counters(
            metrics_url=triton_metrics_url,
            model_name=model_name,
            model_version=model_version,
            timeout=timeout,
        )
        started = time.perf_counter()
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = [
                executor.submit(request_once, client, **request_arguments) for _ in range(requests)
            ]
            results = [future.result() for future in futures]
        elapsed_seconds = time.perf_counter() - started
        triton_after = read_triton_counters(
            metrics_url=triton_metrics_url,
            model_name=model_name,
            model_version=model_version,
            timeout=timeout,
        )

    summary = summarize_service_run(
        results,
        elapsed_seconds=elapsed_seconds,
        concurrency=concurrency,
    )
    summary["triton_metrics"] = summarize_triton_counters(
        triton_before,
        triton_after,
    )
    return summary


def main() -> None:
    args = parse_args()
    if not args.image.is_file():
        raise FileNotFoundError(f"image does not exist: {args.image}")
    if not 0.0 < args.threshold < 1.0:
        raise ValueError("threshold must be between 0 and 1")
    if args.warmup < 0 or args.requests <= 0:
        raise ValueError("warmup must be non-negative and requests must be positive")
    if any(level <= 0 for level in args.concurrency):
        raise ValueError("concurrency levels must be positive")

    base_url = args.base_url.rstrip("/")
    endpoint = f"{base_url}/v1/segment"
    with httpx.Client(timeout=args.timeout, trust_env=False) as readiness_client:
        readiness = readiness_client.get(f"{base_url}/health/ready")
        readiness.raise_for_status()
        if readiness.json().get("status") != "ready":
            raise RuntimeError(f"gateway is not ready: {readiness.text}")

    image_bytes = args.image.read_bytes()
    media_type = mimetypes.guess_type(args.image.name)[0] or "application/octet-stream"
    results = []
    for concurrency in args.concurrency:
        print(f"benchmarking concurrency={concurrency}...")
        result = run_level(
            endpoint=endpoint,
            image_name=args.image.name,
            image_bytes=image_bytes,
            media_type=media_type,
            threshold=args.threshold,
            concurrency=concurrency,
            warmup=args.warmup,
            requests=args.requests,
            timeout=args.timeout,
            triton_metrics_url=args.triton_metrics_url,
            model_name=args.model_name,
            model_version=args.model_version,
        )
        results.append(result)
        print(
            f"qps={result['successful_qps']:.2f} "
            f"p95={result['client_latency']['p95_ms']:.2f}ms "
            f"errors={result['failed_requests']}"
        )

    report = {
        "status": "pass" if all(item["failed_requests"] == 0 for item in results) else "fail",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "endpoint": endpoint,
        "triton_metrics_url": args.triton_metrics_url,
        "model_name": args.model_name,
        "model_version": args.model_version,
        "image": str(args.image),
        "image_sha256": sha256_file(args.image),
        "threshold": args.threshold,
        "warmup_requests_per_level": args.warmup,
        "requests_per_level": args.requests,
        "timing_boundary": ("client request serialization through complete JSON response transfer"),
        "environment_proxy": "ignored by benchmark clients",
        "system": system_metadata(),
        "results": results,
    }
    write_text_atomic(args.report, json.dumps(report, indent=2, sort_keys=True))
    write_text_atomic(args.markdown, render_service_benchmark_markdown(report))
    print("JSON report:", args.report)
    print("Markdown report:", args.markdown)


if __name__ == "__main__":
    main()
