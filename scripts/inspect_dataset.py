from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from industrial_defect.config import load_project_config  # noqa: E402
from industrial_defect.dataset import inspect_severstal_dataset  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate raw Severstal competition data.")
    parser.add_argument("--config", type=Path, default=Path("configs/project.yaml"))
    parser.add_argument("--data-root", type=Path, default=Path("data/raw/severstal"))
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_project_config(args.config)
    report = inspect_severstal_dataset(args.data_root, config.dataset)
    payload = report.to_dict()

    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"saved: {args.output}")

    return 0 if report.valid else 1


if __name__ == "__main__":
    raise SystemExit(main())
