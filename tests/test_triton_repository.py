import hashlib
from pathlib import Path

import pytest

from industrial_defect.triton_repository import prepare_model_repository


def test_prepare_model_repository_copies_engine_and_records_hash(tmp_path: Path) -> None:
    engine = tmp_path / "source.plan"
    engine.write_bytes(b"serialized-engine")
    repository = tmp_path / "repository"
    model_dir = repository / "steel_defect_segmentation"
    model_dir.mkdir(parents=True)
    (model_dir / "config.pbtxt").write_text("name: \"steel_defect_segmentation\"\n")

    manifest = prepare_model_repository(engine, repository)

    destination = model_dir / "1" / "model.plan"
    assert destination.read_bytes() == b"serialized-engine"
    assert manifest["engine_sha256"] == hashlib.sha256(b"serialized-engine").hexdigest()
    assert (model_dir / "1" / "artifact.json").is_file()


def test_prepare_model_repository_requires_engine(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="engine does not exist"):
        prepare_model_repository(tmp_path / "missing.plan", tmp_path / "repository")


def test_prepare_model_repository_requires_positive_version(tmp_path: Path) -> None:
    engine = tmp_path / "source.plan"
    engine.write_bytes(b"engine")
    repository = tmp_path / "repository"
    model_dir = repository / "steel_defect_segmentation"
    model_dir.mkdir(parents=True)
    (model_dir / "config.pbtxt").write_text("name: \"steel_defect_segmentation\"\n")

    with pytest.raises(ValueError, match="version must be positive"):
        prepare_model_repository(engine, repository, version=0)
