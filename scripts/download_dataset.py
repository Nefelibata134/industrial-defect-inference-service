from __future__ import annotations

import argparse
import shutil
import subprocess
import zipfile
from pathlib import Path

COMPETITION = "severstal-steel-defect-detection"
ARCHIVE_NAME = f"{COMPETITION}.zip"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download and extract the Severstal competition dataset."
    )
    parser.add_argument(
        "--download-dir",
        type=Path,
        default=Path("data/downloads"),
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data/raw/severstal"),
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Extract an existing archive without calling the Kaggle CLI.",
    )
    return parser.parse_args()


def safe_extract(archive_path: Path, destination: Path) -> None:
    destination_resolved = destination.resolve()
    with zipfile.ZipFile(archive_path) as archive:
        for member in archive.infolist():
            target = (destination / member.filename).resolve()
            if not target.is_relative_to(destination_resolved):
                raise ValueError(f"unsafe archive path: {member.filename}")
        archive.extractall(destination)


def main() -> None:
    args = parse_args()
    args.download_dir.mkdir(parents=True, exist_ok=True)
    args.data_root.mkdir(parents=True, exist_ok=True)
    archive_path = args.download_dir / ARCHIVE_NAME

    if not args.skip_download:
        if shutil.which("kaggle") is None:
            raise RuntimeError('Kaggle CLI is unavailable; install the project with ".[data]"')
        subprocess.run(
            [
                "kaggle",
                "competitions",
                "download",
                "-c",
                COMPETITION,
                "-p",
                str(args.download_dir),
            ],
            check=True,
        )

    if not archive_path.is_file():
        raise FileNotFoundError(f"dataset archive not found: {archive_path}")

    with zipfile.ZipFile(archive_path) as archive:
        bad_member = archive.testzip()
    if bad_member is not None:
        raise ValueError(f"archive integrity check failed at: {bad_member}")

    safe_extract(archive_path, args.data_root)
    annotation_path = args.data_root / "train.csv"
    images_dir = args.data_root / "train_images"
    if not annotation_path.is_file() or not images_dir.is_dir():
        raise RuntimeError("extracted dataset is missing train.csv or train_images/")

    image_count = sum(
        path.is_file() and path.suffix.lower() in {".jpeg", ".jpg", ".png"}
        for path in images_dir.iterdir()
    )
    print(f"archive: {archive_path}")
    print(f"data root: {args.data_root}")
    print(f"training images: {image_count}")


if __name__ == "__main__":
    main()
