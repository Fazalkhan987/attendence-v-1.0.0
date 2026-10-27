"""Plain data types shared across modules (framework-agnostic, JSON-friendly)."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np


@dataclass(eq=False)
class DetectedFace:
    bbox: np.ndarray                 # float32 [x1, y1, x2, y2] in frame pixels
    score: float                     # detector confidence
    landmarks: np.ndarray | None     # (5, 2): left eye, right eye, nose, left mouth, right mouth

    @property
    def width(self) -> float:
        return float(self.bbox[2] - self.bbox[0])

    @property
    def height(self) -> float:
        return float(self.bbox[3] - self.bbox[1])

    @property
    def size(self) -> float:
        return min(self.width, self.height)

    def to_dict(self) -> dict[str, Any]:
        return {
            "bbox": [round(float(v), 1) for v in self.bbox],
            "score": round(self.score, 4),
            "landmarks": None if self.landmarks is None else self.landmarks.round(1).tolist(),
        }


@dataclass
class QualityReport:
    passed: bool
    reasons: list[str] = field(default_factory=list)
    metrics: dict[str, float] = field(default_factory=dict)


@dataclass(eq=False)
class AnalyzedFace:
    face: DetectedFace
    aligned: np.ndarray | None       # aligned crop (model input size), BGR uint8
    quality: QualityReport
    embedding: np.ndarray | None     # L2-normalised float32, None if not computed


class MatchStatus(str, Enum):
    MATCH = "match"
    UNKNOWN = "unknown"
    LOW_QUALITY = "low_quality"


@dataclass
class MatchResult:
    status: MatchStatus
    person_id: str | None = None
    full_name: str | None = None
    similarity: float | None = None          # best identity score
    runner_up_similarity: float | None = None
    detection_score: float | None = None
    quality_reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status.value,
            "person_id": self.person_id,
            "full_name": self.full_name,
            "similarity": None if self.similarity is None else round(self.similarity, 4),
            "runner_up_similarity": None if self.runner_up_similarity is None
            else round(self.runner_up_similarity, 4),
            "detection_score": None if self.detection_score is None else round(self.detection_score, 4),
            "quality_reasons": self.quality_reasons,
        }


@dataclass
class FaceRecognition:
    """One face in a frame together with its recognition outcome."""
    face: DetectedFace
    result: MatchResult

    def to_dict(self) -> dict[str, Any]:
        return {"face": self.face.to_dict(), "result": self.result.to_dict()}
