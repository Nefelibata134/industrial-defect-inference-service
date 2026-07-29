import asyncio

import cv2
import numpy as np
from fastapi.testclient import TestClient

from industrial_defect.api import create_app
from industrial_defect.serving import ServiceSettings


class FakeModelClient:
    def __init__(self, *, ready: bool = True) -> None:
        self.is_ready = ready

    async def ready(self) -> bool:
        return self.is_ready

    async def infer(
        self,
        images: np.ndarray,
    ) -> tuple[np.ndarray, float]:
        assert images.shape == (1, 3, 256, 1024)
        logits = np.full((1, 4, 256, 1024), -10.0, dtype=np.float32)
        logits[0, 2, 10:20, 20:40] = 10.0
        return logits, 2.5

    async def close(self) -> None:
        return None


class SlowModelClient(FakeModelClient):
    async def ready(self) -> bool:
        await asyncio.sleep(0.05)
        return True

    async def infer(
        self,
        images: np.ndarray,
    ) -> tuple[np.ndarray, float]:
        await asyncio.sleep(0.05)
        return await super().infer(images)


def jpeg_payload(width: int = 1600, height: int = 256) -> bytes:
    image = np.zeros((height, width, 3), dtype=np.uint8)
    success, encoded = cv2.imencode(".jpg", image)
    assert success
    return encoded.tobytes()


def test_health_endpoints() -> None:
    app = create_app(model_client=FakeModelClient())

    with TestClient(app) as client:
        assert client.get("/health/live").json() == {"status": "live"}
        ready = client.get("/health/ready")

    assert ready.status_code == 200
    assert ready.json()["model"] == "steel_defect_segmentation"


def test_segment_returns_encoded_defect_masks() -> None:
    app = create_app(model_client=FakeModelClient())

    with TestClient(app) as client:
        response = client.post(
            "/v1/segment",
            files={"image": ("steel.jpg", jpeg_payload(), "image/jpeg")},
        )

    assert response.status_code == 200
    body = response.json()
    assert body["has_defect"] is True
    assert body["classes"][2]["detected"] is True
    assert body["classes"][2]["rle"]
    assert body["timing_ms"]["inference"] == 2.5


def test_segment_rejects_invalid_dimensions() -> None:
    app = create_app(model_client=FakeModelClient())

    with TestClient(app) as client:
        response = client.post(
            "/v1/segment",
            files={"image": ("small.jpg", jpeg_payload(100, 100), "image/jpeg")},
        )

    assert response.status_code == 422
    assert "image must have shape" in response.json()["detail"]


def test_metrics_endpoint_exposes_request_counter() -> None:
    app = create_app(
        settings=ServiceSettings(),
        model_client=FakeModelClient(),
    )

    with TestClient(app) as client:
        client.get("/health/live")
        response = client.get("/metrics")

    assert response.status_code == 200
    assert "gateway_http_requests_total" in response.text


def test_model_calls_return_503_after_configured_timeout() -> None:
    app = create_app(
        settings=ServiceSettings(request_timeout_seconds=0.01),
        model_client=SlowModelClient(),
    )

    with TestClient(app) as client:
        ready = client.get("/health/ready")
        segment = client.post(
            "/v1/segment",
            files={"image": ("steel.jpg", jpeg_payload(), "image/jpeg")},
        )

    assert ready.status_code == 503
    assert ready.json()["detail"] == "model server readiness check failed"
    assert segment.status_code == 503
    assert segment.json()["detail"] == "model inference failed"
