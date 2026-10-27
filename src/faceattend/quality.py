"""Face quality assessment: size, detector confidence, blur, exposure, contrast, pose.

Pose is a geometric heuristic from the 5 SCRFD landmarks, not a calibrated head-pose
estimator. It is good enough to reject clearly profile/tilted faces at enrollment.
"""
from __future__ import annotations

import math

import cv2
import numpy as np

from .config import QualityConfig
from .schemas import DetectedFace, QualityReport


def pose_metrics(landmarks: np.ndarray) -> dict[str, float]:
    """Return roll (deg), yaw deviation (0 = frontal) and pitch ratio from 5 landmarks."""
    le, re, nose, lm, rm = (np.asarray(p, dtype=np.float64) for p in landmarks)
    dx, dy = re - le
    roll = math.degrees(math.atan2(dy, dx))
    # De-rotate around the eye centre so yaw/pitch are measured in the face frame.
    c = (le + re) / 2.0
    t = -math.radians(roll)
    rot = np.array([[math.cos(t), -math.sin(t)], [math.sin(t), math.cos(t)]])
    le_r, re_r, nose_r, lm_r, rm_r = ((rot @ (p - c)) for p in (le, re, nose, lm, rm))
    eye_span = re_r[0] - le_r[0]
    if eye_span <= 1e-6:
        return {"roll_deg": roll, "yaw_deviation": 1.0, "pitch_ratio": 0.0}
    yaw_ratio = (nose_r[0] - le_r[0]) / eye_span          # ~0.5 when frontal
    mouth_y = (lm_r[1] + rm_r[1]) / 2.0
    pitch_ratio = nose_r[1] / mouth_y if mouth_y > 1e-6 else 0.0   # eye line is y=0
    return {"roll_deg": roll, "yaw_deviation": abs(yaw_ratio - 0.5), "pitch_ratio": float(pitch_ratio)}


class QualityAssessor:
    def __init__(self, cfg: QualityConfig):
        self.cfg = cfg

    def assess(self, face: DetectedFace, aligned: np.ndarray | None) -> QualityReport:
        c = self.cfg
        reasons: list[str] = []
        m: dict[str, float] = {"face_size": round(face.size, 1), "det_score": round(face.score, 4)}

        if face.size < c.min_face_size:
            reasons.append(f"face too small ({face.size:.0f}px < {c.min_face_size}px)")
        if face.score < c.min_det_score:
            reasons.append(f"low detection confidence ({face.score:.2f})")

        if face.landmarks is None:
            reasons.append("no landmarks (cannot align)")
        else:
            p = pose_metrics(face.landmarks)
            m.update({k: round(v, 3) for k, v in p.items()})
            if abs(p["roll_deg"]) > c.max_roll_degrees:
                reasons.append(f"head tilted ({p['roll_deg']:.0f} deg)")
            if p["yaw_deviation"] > c.max_yaw_deviation:
                reasons.append("face turned sideways")
            lo, hi = c.pitch_range
            if not lo <= p["pitch_ratio"] <= hi:
                reasons.append("face tilted up/down")

        if aligned is not None:
            gray = cv2.cvtColor(aligned, cv2.COLOR_BGR2GRAY)
            # Central region avoids background/hair dominating the statistics.
            h, w = gray.shape
            core = gray[h // 8: h - h // 8, w // 8: w - w // 8]
            blur = float(cv2.Laplacian(core, cv2.CV_64F).var())
            bright = float(core.mean())
            contrast = float(core.std())
            m.update(blur_variance=round(blur, 1), brightness=round(bright, 1), contrast=round(contrast, 1))
            if blur < c.min_blur_variance:
                reasons.append("image blurry")
            if bright < c.min_brightness:
                reasons.append("too dark")
            elif bright > c.max_brightness:
                reasons.append("overexposed")
            if contrast < c.min_contrast:
                reasons.append("low contrast")

        return QualityReport(passed=not reasons, reasons=reasons, metrics=m)
