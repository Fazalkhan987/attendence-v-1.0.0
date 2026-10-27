"""Detector interface. Any detector returning DetectedFace objects can be plugged in."""
from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

from ..schemas import DetectedFace


@runtime_checkable
class FaceDetector(Protocol):
    name: str

    def detect(self, image: np.ndarray) -> list[DetectedFace]:
        """Return all faces above the configured confidence threshold, highest score first."""
        ...
