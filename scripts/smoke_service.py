from __future__ import annotations

import argparse
import json
import mimetypes
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from uuid import uuid4


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate the live gateway and run one segmentation request."
    )
    parser.add_argument(
        "--base-url",
        default="http://localhost:8080",
        help="FastAPI gateway base URL.",
    )
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.80)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/reports/service_smoke_response.json"),
    )
    return parser.parse_args()


def read_json(request: Request, *, timeout: float = 30.0) -> dict[str, object]:
    try:
        with urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except HTTPError as error:
        body = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {error.code}: {body}") from error
    except URLError as error:
        raise RuntimeError(f"service request failed: {error.reason}") from error


def multipart_image(image_path: Path) -> tuple[bytes, str]:
    if not image_path.is_file():
        raise FileNotFoundError(f"image does not exist: {image_path}")

    boundary = f"industrial-defect-{uuid4().hex}"
    media_type = mimetypes.guess_type(image_path.name)[0] or "application/octet-stream"
    image_bytes = image_path.read_bytes()
    body = b"".join(
        (
            f"--{boundary}\r\n".encode(),
            (
                'Content-Disposition: form-data; name="image"; '
                f'filename="{image_path.name}"\r\n'
            ).encode(),
            f"Content-Type: {media_type}\r\n\r\n".encode(),
            image_bytes,
            f"\r\n--{boundary}--\r\n".encode(),
        )
    )
    return body, f"multipart/form-data; boundary={boundary}"


def validate_response(payload: dict[str, object]) -> None:
    required = {"model", "image", "threshold", "has_defect", "classes", "timing_ms"}
    missing = sorted(required - payload.keys())
    if missing:
        raise ValueError(f"response is missing fields: {missing}")

    classes = payload["classes"]
    if not isinstance(classes, list) or len(classes) != 4:
        raise ValueError("response must contain four class results")


def main() -> None:
    args = parse_args()
    base_url = args.base_url.rstrip("/")
    readiness = read_json(Request(f"{base_url}/health/ready"))
    if readiness.get("status") != "ready":
        raise RuntimeError(f"gateway is not ready: {readiness}")

    body, content_type = multipart_image(args.image)
    request = Request(
        f"{base_url}/v1/segment?threshold={args.threshold}",
        data=body,
        headers={"Content-Type": content_type},
        method="POST",
    )
    result = read_json(request)
    validate_response(result)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    detected_classes = [
        item["class_name"]
        for item in result["classes"]
        if isinstance(item, dict) and item.get("detected")
    ]
    print("readiness:", readiness)
    print("detected classes:", detected_classes)
    print("timing_ms:", result["timing_ms"])
    print("saved:", args.output)


if __name__ == "__main__":
    main()
