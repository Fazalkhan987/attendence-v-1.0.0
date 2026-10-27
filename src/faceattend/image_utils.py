"""Image loading/validation helpers (unicode-path safe on Windows)."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .exceptions import InvalidImageError

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
MAX_IMAGE_SIDE = 8000


def validate_frame(image: object) -> np.ndarray:
    """Raise InvalidImageError unless ``image`` is a usable BGR uint8 frame."""
    if not isinstance(image, np.ndarray):
        raise InvalidImageError("Frame is not a numpy array")
    if image.ndim != 3 or image.shape[2] != 3:
        raise InvalidImageError(f"Expected HxWx3 BGR image, got shape {image.shape}")
    if image.dtype != np.uint8:
        raise InvalidImageError(f"Expected uint8 image, got {image.dtype}")
    h, w = image.shape[:2]
    if h < 32 or w < 32:
        raise InvalidImageError(f"Image too small: {w}x{h}")
    if max(h, w) > MAX_IMAGE_SIDE:
        raise InvalidImageError(f"Image too large: {w}x{h}")
    return image


def read_image(path: str | Path) -> np.ndarray:
    p = Path(path)
    if not p.is_file():
        raise InvalidImageError(f"File not found: {p}")
    data = np.fromfile(str(p), dtype=np.uint8)
    if data.size == 0:
        raise InvalidImageError(f"Empty file: {p}")
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise InvalidImageError(f"Corrupted or unsupported image: {p.name}")
    return validate_frame(img)


def write_image(path: str | Path, image: np.ndarray) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(p.suffix or ".png", image)
    if not ok:
        raise InvalidImageError(f"Could not encode image for {p}")
    buf.tofile(str(p))


def list_images(directory: str | Path) -> list[Path]:
    d = Path(directory)
    if not d.is_dir():
        raise InvalidImageError(f"Not a directory: {d}")
    return sorted(p for p in d.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)
