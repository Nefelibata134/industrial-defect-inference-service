from __future__ import annotations

from time import perf_counter

import numpy as np
from numpy.typing import NDArray


class TritonHttpModelClient:
    def __init__(
        self,
        url: str,
        *,
        model_name: str,
        model_version: str,
    ) -> None:
        try:
            from tritonclient.http import aio as httpclient
        except ImportError as error:
            raise RuntimeError(
                "Triton HTTP client is not installed; install the service dependencies"
            ) from error

        self._httpclient = httpclient
        self._client = httpclient.InferenceServerClient(url=url, verbose=False)
        self._model_name = model_name
        self._model_version = model_version

    async def ready(self) -> bool:
        server_live = await self._client.is_server_live()
        server_ready = await self._client.is_server_ready()
        model_ready = await self._client.is_model_ready(
            self._model_name,
            self._model_version,
        )
        return bool(server_live and server_ready and model_ready)

    async def infer(
        self,
        images: NDArray[np.float32],
    ) -> tuple[NDArray[np.float32], float]:
        if images.dtype != np.float32:
            raise ValueError(f"images must have dtype float32, got {images.dtype}")
        if images.ndim != 4:
            raise ValueError(f"images must have shape N x C x H x W, got {images.shape}")

        infer_input = self._httpclient.InferInput("images", images.shape, "FP32")
        infer_input.set_data_from_numpy(images, binary_data=True)
        requested_output = self._httpclient.InferRequestedOutput(
            "logits",
            binary_data=True,
        )

        started = perf_counter()
        result = await self._client.infer(
            model_name=self._model_name,
            model_version=self._model_version,
            inputs=[infer_input],
            outputs=[requested_output],
        )
        elapsed_ms = (perf_counter() - started) * 1000.0
        logits = result.as_numpy("logits")
        if logits is None:
            raise RuntimeError("Triton response does not contain logits")
        return np.asarray(logits, dtype=np.float32), elapsed_ms

    async def close(self) -> None:
        await self._client.close()
