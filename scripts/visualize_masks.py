from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from industrial_defect.config import load_project_config  # noqa: E402
from industrial_defect.rle import decode_rle  # noqa: E402

COLORS = (
    (40, 40, 230),
    (40, 200, 40),
    (230, 150, 30),
    (200, 40, 180),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render Severstal RLE mask overlays.")
    parser.add_argument("--config", type=Path, default=Path("configs/project.yaml"))
    parser.add_argument("--data-root", type=Path, default=Path("data/raw/severstal"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/mask_samples"),
    )
    parser.add_argument("--image-id", action="append", default=[])
    parser.add_argument("--count", type=int, default=4)
    parser.add_argument("--alpha", type=float, default=0.45)
    return parser.parse_args()


def read_positive_masks(annotation_path: Path) -> dict[str, dict[int, str]]:
    masks: defaultdict[str, dict[int, str]] = defaultdict(dict)
    with annotation_path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        columns = set(reader.fieldnames or [])
        combined = {"ImageId_ClassId", "EncodedPixels"} <= columns
        separate = {"ImageId", "ClassId", "EncodedPixels"} <= columns
        if not (combined or separate):
            raise ValueError("unsupported annotation columns")

        for line_number, row in enumerate(reader, start=2):
            if combined:
                identity = (row.get("ImageId_ClassId") or "").strip()
                try:
                    image_id, class_id_text = identity.rsplit("_", 1)
                    class_id = int(class_id_text)
                except ValueError as error:
                    raise ValueError(f"train.csv:{line_number}: invalid ImageId_ClassId") from error
            else:
                image_id = (row.get("ImageId") or "").strip()
                class_id = int((row.get("ClassId") or "").strip())

            encoded_pixels = (row.get("EncodedPixels") or "").strip()
            if encoded_pixels:
                masks[image_id][class_id] = encoded_pixels
    return dict(masks)


def choose_examples(
    masks: dict[str, dict[int, str]],
    requested: list[str],
    count: int,
    class_count: int,
) -> list[str]:
    if requested:
        missing = [image_id for image_id in requested if image_id not in masks]
        if missing:
            raise ValueError(f"requested images have no positive masks: {missing}")
        return requested

    selected: list[str] = []
    for class_id in range(1, class_count + 1):
        image_id = next(
            (
                candidate
                for candidate in sorted(masks)
                if class_id in masks[candidate] and candidate not in selected
            ),
            None,
        )
        if image_id:
            selected.append(image_id)

    for image_id in sorted(masks):
        if len(selected) >= count:
            break
        if image_id not in selected:
            selected.append(image_id)
    return selected[:count]


def render_overlay(
    image: np.ndarray,
    encoded_by_class: dict[int, str],
    alpha: float,
) -> tuple[np.ndarray, dict[int, int]]:
    overlay = image.copy()
    areas: dict[int, int] = {}
    for class_id, encoded_pixels in sorted(encoded_by_class.items()):
        mask = decode_rle(encoded_pixels, shape=image.shape[:2]).astype(bool)
        areas[class_id] = int(mask.sum())
        color = np.asarray(COLORS[class_id - 1], dtype=np.float32)
        overlay[mask] = ((1.0 - alpha) * overlay[mask].astype(np.float32) + alpha * color).astype(
            np.uint8
        )
    return overlay, areas


def main() -> None:
    args = parse_args()
    if not 0.0 < args.alpha <= 1.0:
        raise ValueError("alpha must be in the interval (0, 1]")
    if args.count <= 0:
        raise ValueError("count must be positive")

    config = load_project_config(args.config)
    masks = read_positive_masks(args.data_root / "train.csv")
    image_ids = choose_examples(
        masks,
        requested=args.image_id,
        count=args.count,
        class_count=len(config.dataset.classes),
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for image_id in image_ids:
        image_path = args.data_root / "train_images" / image_id
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"failed to read image: {image_path}")

        overlay, areas = render_overlay(image, masks[image_id], args.alpha)
        output_path = args.output_dir / f"{Path(image_id).stem}_overlay.jpg"
        if not cv2.imwrite(str(output_path), overlay):
            raise RuntimeError(f"failed to save image: {output_path}")

        labels = ", ".join(
            f"{config.dataset.classes[class_id - 1]}={area}px"
            for class_id, area in sorted(areas.items())
        )
        print(f"{image_id}: {labels}")
        print(f"saved: {output_path}")


if __name__ == "__main__":
    main()
