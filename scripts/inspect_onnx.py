from __future__ import annotations

import argparse
import json

from industrial_defect.config import load_project_config
from industrial_defect.onnx_export import summarize_onnx_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate and inspect an ONNX model.")
    parser.add_argument("--config", default="configs/project.yaml")
    parser.add_argument("--model")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_project_config(args.config)
    model_path = args.model or config.deployment.onnx_model
    print(json.dumps(summarize_onnx_model(model_path), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
