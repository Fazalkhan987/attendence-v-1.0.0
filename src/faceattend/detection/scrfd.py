"""SCRFD detector (InsightFace model zoo) behind the FaceDetector interface."""
from __future__ import annotations

import numpy as np

from ..image_utils import validate_frame
from ..schemas import DetectedFace


class SCRFDDetector:
    """Wraps an already-prepared ``insightface.model_zoo.scrfd.SCRFD`` instance."""

    def __init__(self, model, confidence_threshold: float = 0.5, max_faces: int = 0,
                 nms_threshold: float | None = None, name: str = "scrfd"):
        self._model = model
        self.confidence_threshold = float(confidence_threshold)
        self.max_faces = int(max_faces)
        self.name = name
        # SCRFD applies its own det_thresh internally; keep it in sync so we never
        # drop faces before our own threshold sees them.
        self._model.det_thresh = self.confidence_threshold
        if nms_threshold is not None:
            self._model.nms_thresh = float(nms_threshold)

    def detect(self, image: np.ndarray) -> list[DetectedFace]:
        validate_frame(image)
        bboxes, kpss = self._model.detect(image, max_num=self.max_faces, metric="default")
        faces: list[DetectedFace] = []
        if bboxes is None or len(bboxes) == 0:
            return faces
        h, w = image.shape[:2]
        for i in range(bboxes.shape[0]):
            score = float(bboxes[i, 4])
            if score < self.confidence_threshold:
                continue
            box = bboxes[i, :4].astype(np.float32)
            box[[0, 2]] = np.clip(box[[0, 2]], 0, w - 1)
            box[[1, 3]] = np.clip(box[[1, 3]], 0, h - 1)
            if box[2] - box[0] < 2 or box[3] - box[1] < 2:
                continue
            kps = None if kpss is None else kpss[i].astype(np.float32)
            faces.append(DetectedFace(bbox=box, score=score, landmarks=kps))
        faces.sort(key=lambda f: f.score, reverse=True)
        return faces
