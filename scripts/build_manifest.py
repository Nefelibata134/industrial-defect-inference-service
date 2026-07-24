from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from industrial_defect.config import load_project_config  # noqa: E402
from industrial_defect.manifest import (  # noqa: E402
    ManifestRow,
    build_split_manifest,
    read_image_labels,
    write_manifest_csv,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a deterministic multilabel Severstal split manifest."
    )
    parser.add_argument("--config", type=Path, default=Path("configs/project.yaml"))
    parser.add_argument("--data-root", type=Path, default=Path("data/raw/severstal"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/manifests/severstal_v1.csv"),
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("outputs/reports/split_summary.json"),
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def summarize(
    rows: list[ManifestRow],
    split_names: tuple[str, ...],
    class_names: tuple[str, ...],
) -> dict[str, object]:
    split_sizes = Counter(row.split for row in rows)
    split_positive_images = Counter(row.split for row in rows if row.has_defect)
    label_sets = Counter("".join(str(value) for value in row.labels) for row in rows)
    per_split_classes = {
        split: {
            class_name: sum(row.labels[class_index] for row in rows if row.split == split)
            for class_index, class_name in enumerate(class_names)
        }
        for split in split_names
    }

    return {
        "total_images": len(rows),
        "positive_images": sum(row.has_defect for row in rows),
        "normal_images": sum(not row.has_defect for row in rows),
        "split_sizes": dict(split_sizes),
        "split_positive_images": dict(split_positive_images),
        "class_positive_images": {
            class_name: sum(row.labels[class_index] for row in rows)
            for class_index, class_name in enumerate(class_names)
        },
        "per_split_class_positive_images": per_split_classes,
        "label_set_distribution": dict(sorted(label_sets.items())),
    }


def main() -> None:
    args = parse_args()
    config = load_project_config(args.config)
    annotation_path = args.data_root / "train.csv"
    images_dir = args.data_root / "train_images"
    image_ids = sorted(
        path.name
        for path in images_dir.iterdir()
        if path.is_file() and path.suffix.lower() in {".jpeg", ".jpg", ".png"}
    )
    records = read_image_labels(
        annotation_path,
        len(config.dataset.classes),
        image_ids=image_ids,
    )
    rows = build_split_manifest(
        records,
        split_names=config.dataset.splits,
        ratios=config.dataset.split_ratio,
        seed=config.seed,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        write_manifest_csv(rows, config.dataset.classes, handle)

    summary = summarize(rows, config.dataset.splits, config.dataset.classes)
    summary.update(
        {
            "seed": config.seed,
            "split_ratio": dict(
                zip(
                    config.dataset.splits,
                    config.dataset.split_ratio,
                    strict=True,
                )
            ),
            "annotation_sha256": sha256_file(annotation_path),
            "manifest_sha256": sha256_file(args.output),
        }
    )
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"manifest: {args.output}")
    print(f"summary: {args.summary}")


if __name__ == "__main__":
    main()
