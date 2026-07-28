from __future__ import annotations

import argparse
import json
from pathlib import Path

from industrial_defect.triton_repository import prepare_model_repository

DEFAULT_ENGINE = Path("models/unet_resnet18_severstal_fp16.plan")
DEFAULT_REPOSITORY = Path("model_repository")
DEFAULT_MODEL_NAME = "steel_defect_segmentation"
DEFAULT_VERSION = 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Materialize a versioned Triton model repository.",
    )
    parser.add_argument("--engine", type=Path, default=DEFAULT_ENGINE)
    parser.add_argument("--repository", type=Path, default=DEFAULT_REPOSITORY)
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--version", type=int, default=DEFAULT_VERSION)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = prepare_model_repository(
        args.engine,
        args.repository,
        model_name=args.model_name,
        version=args.version,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
