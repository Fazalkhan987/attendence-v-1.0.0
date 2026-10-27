"""Liveness / presentation-attack detection interface.

STATUS: PLANNED - interface only. No anti-spoofing model is integrated or tested.
A plain RGB webcam + face recognition model CANNOT reliably reject printed photos or
screen replays. Until a validated PAD model is integrated (and evaluated with
APCER/BPCER on representative attacks), every result is ``NOT_CHECKED``.

Integration plan: implement ``LivenessChecker`` with a PAD model (e.g. a
MiniFASNet/CDCN-style ONNX model, licence permitting), run it on the face crop
for confirmed tracks only, and require ``LIVE`` before an attendance event is accepted.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Protocol

import numpy as np

from ..schemas import DetectedFace


class LivenessStatus(str, Enum):
    NOT_CHECKED = "not_checked"
    LIVE = "live"
    SPOOF = "spoof"
    UNCERTAIN = "uncertain"


@dataclass
class LivenessResult:
    status: LivenessStatus
    score: float | None = None


class LivenessChecker(Protocol):
    def check(self, frame: np.ndarray, face: DetectedFace) -> LivenessResult: ...


class NullLivenessChecker:
    """Placeholder that never claims liveness."""

    def check(self, frame: np.ndarray, face: DetectedFace) -> LivenessResult:
        return LivenessResult(LivenessStatus.NOT_CHECKED)
