"""IoU face tracker with temporal vote smoothing.

Recognition runs per track every N frames, and a label is shown only after enough
consistent votes accumulate in a sliding window - suppressing flicker and one-off
false matches.
"""
from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass, field
from enum import Enum

import numpy as np

from ..schemas import DetectedFace, MatchResult, MatchStatus


def iou(a: np.ndarray, b: np.ndarray) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return float(inter / union) if union > 0 else 0.0


class TrackState(str, Enum):
    VERIFYING = "verifying"
    CONFIRMED = "confirmed"
    UNKNOWN = "unknown"
    LOW_QUALITY = "low_quality"


@dataclass
class TrackDecision:
    state: TrackState
    person_id: str | None = None
    full_name: str | None = None
    similarity: float | None = None


@dataclass(eq=False)
class Track:
    track_id: int
    face: DetectedFace
    window: int
    missed: int = 0
    frames_since_recognition: int = 10**9
    history: deque = field(default=None)
    last_result: MatchResult | None = None
    confirmed_person: str | None = None     # person announced for this track (event fired once)

    def __post_init__(self):
        if self.history is None:
            self.history = deque(maxlen=self.window)

    def add(self, result: MatchResult) -> None:
        self.last_result = result
        self.frames_since_recognition = 0
        if result.status != MatchStatus.LOW_QUALITY:
            self.history.append(result)

    def decide(self, min_votes: int) -> TrackDecision:
        votes = [r.person_id if r.status == MatchStatus.MATCH else None for r in self.history]
        if votes:
            pid, n = Counter(votes).most_common(1)[0]
            if pid is not None and n >= min_votes:
                sims = [r.similarity for r in self.history if r.person_id == pid]
                name = next(r.full_name for r in self.history if r.person_id == pid)
                return TrackDecision(TrackState.CONFIRMED, pid, name, float(np.mean(sims)))
            unknown = sum(v is None for v in votes)
            if unknown >= min_votes:
                sims = [r.similarity for r in self.history if r.similarity is not None]
                return TrackDecision(TrackState.UNKNOWN, similarity=max(sims) if sims else None)
        if self.last_result is not None and self.last_result.status == MatchStatus.LOW_QUALITY \
                and len(self.history) < min_votes:
            return TrackDecision(TrackState.LOW_QUALITY)
        return TrackDecision(TrackState.VERIFYING)


class FaceTracker:
    def __init__(self, iou_threshold: float = 0.3, max_missed: int = 10, window: int = 7):
        self.iou_threshold = iou_threshold
        self.max_missed = max_missed
        self.window = window
        self.tracks: list[Track] = []
        self._next_id = 1

    def update(self, faces: list[DetectedFace]) -> list[Track]:
        """Associate detections to tracks (greedy IoU); return tracks visible this frame."""
        pairs = sorted(
            ((iou(t.face.bbox, f.bbox), ti, fi) for ti, t in enumerate(self.tracks)
             for fi, f in enumerate(faces)), reverse=True)
        used_t, used_f, visible = set(), set(), []
        for score, ti, fi in pairs:
            if score < self.iou_threshold:
                break
            if ti in used_t or fi in used_f:
                continue
            t = self.tracks[ti]
            t.face, t.missed = faces[fi], 0
            t.frames_since_recognition += 1
            used_t.add(ti)
            used_f.add(fi)
            visible.append(t)
        for ti, t in enumerate(self.tracks):
            if ti not in used_t:
                t.missed += 1
        for fi, f in enumerate(faces):
            if fi not in used_f:
                t = Track(self._next_id, f, self.window)
                self._next_id += 1
                self.tracks.append(t)
                visible.append(t)
        self.tracks = [t for t in self.tracks if t.missed <= self.max_missed]
        return visible
