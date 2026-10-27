"""Drawing helpers for the webcam window."""
from __future__ import annotations

import cv2
import numpy as np

from .tracker import TrackDecision, TrackState

COLORS = {  # BGR
    TrackState.CONFIRMED: (60, 200, 60),
    TrackState.UNKNOWN: (40, 40, 230),
    TrackState.VERIFYING: (0, 200, 230),
    TrackState.LOW_QUALITY: (0, 140, 255),
}


def draw_label(frame: np.ndarray, x: int, y: int, lines: list[str], color) -> None:
    font, scale, th = cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1
    sizes = [cv2.getTextSize(t, font, scale, th)[0] for t in lines]
    h_line = max(s[1] for s in sizes) + 8
    w = max(s[0] for s in sizes) + 10
    top = max(0, y - h_line * len(lines))
    cv2.rectangle(frame, (x, top), (x + w, top + h_line * len(lines)), color, -1)
    for i, t in enumerate(lines):
        cv2.putText(frame, t, (x + 5, top + h_line * (i + 1) - 5), font, scale, (255, 255, 255),
                    th, cv2.LINE_AA)


def draw_track(frame: np.ndarray, bbox: np.ndarray, det_score: float, d: TrackDecision) -> None:
    x1, y1, x2, y2 = (int(v) for v in bbox)
    color = COLORS[d.state]
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    if d.state == TrackState.CONFIRMED:
        head = f"{d.full_name} ({d.person_id})"
    elif d.state == TrackState.UNKNOWN:
        head = "Unknown"
    elif d.state == TrackState.LOW_QUALITY:
        head = "Low quality - face camera"
    else:
        head = "Verifying..."
    sim = "-" if d.similarity is None else f"{d.similarity:.2f}"
    draw_label(frame, x1, y1, [head, f"det {det_score:.2f} | sim {sim}"], color)


def draw_status(frame: np.ndarray, text: str) -> None:
    cv2.putText(frame, text, (10, frame.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(frame, text, (10, frame.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (255, 255, 255), 1, cv2.LINE_AA)
