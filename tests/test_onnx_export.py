from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from torch import nn

from industrial_defect.onnx_export import (
    export_onnx_model,
    sha256_file,
)


class TinySegmentationModel(nn.Module):
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return images[:, :2] * 2.0


def test_sha256_file(tmp_path: Path) -> None:
    path = tmp_path / "artifact.bin"
    path.write_bytes(b"industrial-defect")

    assert sha256_file(path) == (
        "a3c906d209c6f187fcda09e75e0b8706bf7ec304f123073dcda0e846038a88a1"
    )


def test_exported_model_has_dynamic_batch_and_runs_in_onnxruntime(
    tmp_path: Path,
) -> None:
    model_path = tmp_path / "tiny.onnx"
    summary = export_onnx_model(
        TinySegmentationModel().eval(),
        torch.zeros((1, 3, 8, 8), dtype=torch.float32),
        model_path,
        input_name="images",
        output_name="logits",
        opset=18,
        dynamic_batch=True,
        metadata={"architecture": "tiny", "threshold": "0.80"},
    )

    assert summary["inputs"] == [
        {"name": "images", "shape": ["batch", 3, 8, 8], "dtype": "float"}
    ]
    assert summary["outputs"] == [
        {"name": "logits", "shape": ["batch", 2, 8, 8], "dtype": "float"}
    ]
    assert summary["metadata"]["architecture"] == "tiny"
    assert summary["metadata"]["threshold"] == "0.80"
    assert summary["opsets"]["ai.onnx"] == 18

    session = ort.InferenceSession(
        str(model_path),
        providers=["CPUExecutionProvider"],
    )
    inputs = np.ones((3, 3, 8, 8), dtype=np.float32)
    outputs = session.run(["logits"], {"images": inputs})[0]

    assert outputs.shape == (3, 2, 8, 8)
    np.testing.assert_allclose(outputs, inputs[:, :2] * 2.0)
