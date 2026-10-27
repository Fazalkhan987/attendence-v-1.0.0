"""Webcam wrapper with explicit, actionable errors."""
from __future__ import annotations

import logging
import platform
import time

import cv2
import numpy as np

from ..config import LiveConfig
from ..exceptions import CameraError

log = logging.getLogger(__name__)

_BACKENDS = {
    "dshow": cv2.CAP_DSHOW, "msmf": cv2.CAP_MSMF, "v4l2": cv2.CAP_V4L2,
    "avfoundation": cv2.CAP_AVFOUNDATION,
}


class Webcam:
    def __init__(self, cfg: LiveConfig):
        self.cfg = cfg
        self._cap: cv2.VideoCapture | None = None
        self._failures = 0

    def open(self) -> "Webcam":
        backend = self.cfg.backend.lower()
        if backend == "auto":
            api = cv2.CAP_DSHOW if platform.system() == "Windows" else cv2.CAP_ANY
        elif backend in _BACKENDS:
            api = _BACKENDS[backend]
        else:
            raise CameraError(f"Unknown camera backend '{self.cfg.backend}'")
        cap = cv2.VideoCapture(self.cfg.camera_index, api)
        if not cap.isOpened():
            cap.release()
            raise CameraError(
                f"Cannot open camera index {self.cfg.camera_index}. Check that it is connected, "
                "not used by another app, and that this terminal has camera permission "
                "(macOS: System Settings > Privacy & Security > Camera; Windows: Settings > "
                "Privacy > Camera; Linux: user in 'video' group).")
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.cfg.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.cfg.height)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        ok = False
        for _ in range(20):           # some cameras return empty frames while warming up
            ok, frame = cap.read()
            if ok and frame is not None:
                break
            time.sleep(0.05)
        if not ok:
            cap.release()
            raise CameraError("Camera opened but returned no frames (permission denied or device busy?)")
        self._cap = cap
        log.info("Camera %d opened at %dx%d", self.cfg.camera_index,
                 int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        return self

    def read(self) -> np.ndarray | None:
        """Return a frame, None for a transient failure; raise after too many failures."""
        if self._cap is None:
            raise CameraError("Camera is not open")
        ok, frame = self._cap.read()
        if not ok or frame is None or frame.size == 0:
            self._failures += 1
            if self._failures >= self.cfg.max_consecutive_read_failures:
                raise CameraError("Video stream interrupted (camera disconnected?)")
            time.sleep(0.02)
            return None
        self._failures = 0
        return frame

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None
            log.info("Camera released")

    def __enter__(self) -> "Webcam":
        return self.open()

    def __exit__(self, *exc) -> None:
        self.close()
