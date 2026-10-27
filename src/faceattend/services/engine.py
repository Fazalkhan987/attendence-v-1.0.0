"""FaceEngine: detection + alignment + quality + embedding, independent of any UI.

This is the object a FastAPI app would create once at startup and share.
"""
from __future__ import annotations

import contextlib
import io
import logging
from pathlib import Path

import numpy as np

from ..config import AppConfig
from ..detection.base import FaceDetector
from ..exceptions import InvalidImageError, ModelLoadError
from ..image_utils import validate_frame
from ..quality import QualityAssessor
from ..recognition.base import FaceEmbedder
from ..schemas import AnalyzedFace, DetectedFace

log = logging.getLogger(__name__)


class FaceEngine:
    def __init__(self, detector: FaceDetector, embedder: FaceEmbedder):
        self.detector = detector
        self.embedder = embedder

    @property
    def model_name(self) -> str:
        return self.embedder.name

    def detect(self, image: np.ndarray) -> list[DetectedFace]:
        return self.detector.detect(validate_frame(image))

    def analyze(self, image: np.ndarray, faces: list[DetectedFace], assessor: QualityAssessor,
                embed_failed_quality: bool = False) -> list[AnalyzedFace]:
        """Align, quality-check and (batch-)embed the given faces."""
        results: list[AnalyzedFace] = []
        to_embed: list[int] = []
        for face in faces:
            aligned = None
            if face.landmarks is not None:
                try:
                    aligned = self.embedder.align(image, face)
                except InvalidImageError:
                    aligned = None
            q = assessor.assess(face, aligned)
            results.append(AnalyzedFace(face=face, aligned=aligned, quality=q, embedding=None))
            if aligned is not None and (q.passed or embed_failed_quality):
                to_embed.append(len(results) - 1)
        if to_embed:
            embs = self.embedder.embed([results[i].aligned for i in to_embed])
            for i, e in zip(to_embed, embs):
                results[i].embedding = e
        return results


def build_engine(cfg: AppConfig) -> FaceEngine:
    """Load SCRFD + ArcFace from the configured InsightFace model pack (downloads on first use)."""
    try:
        from insightface.app import FaceAnalysis
    except Exception as exc:  # ImportError or broken native extension
        raise ModelLoadError(f"InsightFace could not be imported: {exc}. See README 'Installation'.") from exc

    from ..detection.scrfd import SCRFDDetector
    from ..recognition.arcface import ArcFaceEmbedder

    m = cfg.model
    root = str(Path(m.root).expanduser())
    # InsightFace prints progress to stdout; capture it so CLI/JSON output stays clean.
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            app = FaceAnalysis(name=m.pack, root=root, allowed_modules=["detection", "recognition"],
                               providers=list(m.providers))
            app.prepare(ctx_id=m.ctx_id, det_thresh=cfg.detection.confidence_threshold,
                        det_size=tuple(m.det_size))
        log.debug("insightface: %s", buf.getvalue().strip())
    except Exception as exc:
        raise ModelLoadError(f"Failed to load model pack '{m.pack}' from {root}: {exc}") from exc
    if "detection" not in app.models or "recognition" not in app.models:
        raise ModelLoadError(f"Model pack '{m.pack}' lacks a detection or recognition model")

    rec = app.models["recognition"]
    detector = SCRFDDetector(app.models["detection"], cfg.detection.confidence_threshold,
                             cfg.detection.max_faces, cfg.detection.nms_threshold)
    embedder = ArcFaceEmbedder(rec, name=f"insightface/{m.pack}/{Path(rec.model_file).name}")
    log.info("Loaded detector=%s embedder=%s providers=%s",
             Path(app.models['detection'].model_file).name, embedder.name,
             rec.session.get_providers())
    return FaceEngine(detector, embedder)
