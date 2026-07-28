from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from time import perf_counter
from typing import Annotated

import numpy as np
from fastapi import FastAPI, File, HTTPException, Query, Request, Response, UploadFile
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Histogram,
    generate_latest,
)

from industrial_defect.inference_data import preprocess_image
from industrial_defect.serving import (
    AsyncModelClient,
    ServiceSettings,
    decode_image,
    serialize_segmentation,
)
from industrial_defect.triton_client import TritonHttpModelClient

ALLOWED_MEDIA_TYPES = {"image/jpeg", "image/png"}


class GatewayMetrics:
    def __init__(self) -> None:
        self.registry = CollectorRegistry()
        self.requests = Counter(
            "gateway_http_requests_total",
            "HTTP requests handled by the inference gateway.",
            ("method", "path", "status"),
            registry=self.registry,
        )
        self.request_duration = Histogram(
            "gateway_http_request_duration_seconds",
            "End-to-end gateway request latency.",
            ("method", "path"),
            registry=self.registry,
        )
        self.inference_duration = Histogram(
            "gateway_model_inference_duration_seconds",
            "Triton request latency observed by the gateway.",
            registry=self.registry,
        )


def create_app(
    *,
    settings: ServiceSettings | None = None,
    model_client: AsyncModelClient | None = None,
) -> FastAPI:
    service_settings = settings or ServiceSettings.from_env()
    metrics = GatewayMetrics()
    owns_model_client = model_client is None

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.model_client = model_client or TritonHttpModelClient(
            service_settings.triton_url,
            model_name=service_settings.model_name,
            model_version=service_settings.model_version,
        )
        try:
            yield
        finally:
            if owns_model_client:
                await app.state.model_client.close()

    app = FastAPI(
        title="Industrial Defect Inference Gateway",
        version="0.4.0",
        lifespan=lifespan,
    )

    @app.middleware("http")
    async def observe_request(request: Request, call_next):
        started = perf_counter()
        status_code = 500
        try:
            response = await call_next(request)
            status_code = response.status_code
            return response
        finally:
            path = request.url.path
            metrics.requests.labels(
                request.method,
                path,
                str(status_code),
            ).inc()
            metrics.request_duration.labels(request.method, path).observe(
                perf_counter() - started
            )

    @app.get("/health/live")
    async def live() -> dict[str, str]:
        return {"status": "live"}

    @app.get("/health/ready")
    async def ready(request: Request) -> dict[str, object]:
        try:
            model_ready = await request.app.state.model_client.ready()
        except Exception as error:
            raise HTTPException(
                status_code=503,
                detail="model server readiness check failed",
            ) from error
        if not model_ready:
            raise HTTPException(status_code=503, detail="model is not ready")
        return {
            "status": "ready",
            "model": service_settings.model_name,
            "version": service_settings.model_version,
        }

    @app.get("/metrics")
    async def prometheus_metrics() -> Response:
        return Response(
            content=generate_latest(metrics.registry),
            media_type=CONTENT_TYPE_LATEST,
        )

    @app.post("/v1/segment")
    async def segment(
        request: Request,
        image: Annotated[UploadFile, File(description="1600 x 256 steel image")],
        threshold: Annotated[float, Query(gt=0.0, lt=1.0)] = service_settings.threshold,
    ) -> dict[str, object]:
        request_started = perf_counter()
        if image.content_type not in ALLOWED_MEDIA_TYPES:
            raise HTTPException(
                status_code=415,
                detail=f"unsupported media type: {image.content_type}",
            )

        payload = await image.read(service_settings.max_upload_bytes + 1)
        if len(payload) > service_settings.max_upload_bytes:
            raise HTTPException(status_code=413, detail="image payload is too large")

        preprocess_started = perf_counter()
        try:
            decoded = decode_image(payload, service_settings)
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        tensor = preprocess_image(
            decoded,
            (service_settings.input_width, service_settings.input_height),
        )[None]
        preprocess_ms = (perf_counter() - preprocess_started) * 1000.0

        try:
            logits, inference_ms = await request.app.state.model_client.infer(tensor)
        except Exception as error:
            raise HTTPException(
                status_code=503,
                detail="model inference failed",
            ) from error
        metrics.inference_duration.observe(inference_ms / 1000.0)

        postprocess_started = perf_counter()
        classes = serialize_segmentation(
            np.asarray(logits, dtype=np.float32),
            threshold=threshold,
            output_size=(
                service_settings.source_width,
                service_settings.source_height,
            ),
        )
        postprocess_ms = (perf_counter() - postprocess_started) * 1000.0

        return {
            "model": {
                "name": service_settings.model_name,
                "version": service_settings.model_version,
                "backend": "tensorrt",
            },
            "image": {
                "filename": image.filename,
                "width": service_settings.source_width,
                "height": service_settings.source_height,
            },
            "threshold": threshold,
            "has_defect": any(item["detected"] for item in classes),
            "classes": classes,
            "timing_ms": {
                "preprocess": round(preprocess_ms, 3),
                "inference": round(inference_ms, 3),
                "postprocess": round(postprocess_ms, 3),
                "total": round((perf_counter() - request_started) * 1000.0, 3),
            },
        }

    return app


app = create_app()
