from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

DEFAULT_MODEL_NAME = "steel_defect_segmentation"
DEFAULT_VERSION = 1


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare_model_repository(
    engine_path: str | Path,
    repository_path: str | Path,
    *,
    model_name: str = DEFAULT_MODEL_NAME,
    version: int = DEFAULT_VERSION,
) -> dict[str, object]:
    source = Path(engine_path)
    repository = Path(repository_path)
    config_path = repository / model_name / "config.pbtxt"
    version_dir = repository / model_name / str(version)

    if not source.is_file():
        raise FileNotFoundError(f"TensorRT engine does not exist: {source}")
    if not config_path.is_file():
        raise FileNotFoundError(f"Triton model config does not exist: {config_path}")
    if version <= 0:
        raise ValueError("model version must be positive")

    version_dir.mkdir(parents=True, exist_ok=True)
    destination = version_dir / "model.plan"
    shutil.copy2(source, destination)

    manifest = {
        "model_name": model_name,
        "model_version": version,
        "source_engine": str(source),
        "repository_engine": str(destination),
        "engine_bytes": destination.stat().st_size,
        "engine_sha256": sha256_file(destination),
    }
    manifest_path = version_dir / "artifact.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest
