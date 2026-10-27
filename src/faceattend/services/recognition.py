"""Frame-level recognition service (stateless per call; usable from webcam loop or an API)."""
from __future__ import annotations

import logging
import threading

import numpy as np

from ..config import RecognitionConfig
from ..quality import QualityAssessor
from ..recognition.matcher import EmbeddingGallery, FaceMatcher
from ..schemas import AnalyzedFace, DetectedFace, FaceRecognition, MatchResult, MatchStatus
from ..storage.database import Repository
from .engine import FaceEngine

log = logging.getLogger(__name__)


class RecognitionService:
    def __init__(self, engine: FaceEngine, repo: Repository, cfg: RecognitionConfig,
                 assessor: QualityAssessor):
        self.engine = engine
        self.repo = repo
        self.cfg = cfg
        self.assessor = assessor
        self._lock = threading.RLock()
        self._gallery = EmbeddingGallery(cfg.match_strategy, cfg.top_k)
        self._matcher = FaceMatcher(self._gallery, cfg.similarity_threshold, cfg.margin)
        self._names: dict[str, str] = {}
        self.reload_gallery()

    def reload_gallery(self) -> int:
        """Re-read embeddings from the DB (call after any registration change)."""
        with self._lock:
            self._gallery.load(self.repo.load_embeddings(self.engine.model_name))
            self._names = self.repo.names()
            log.info("Gallery loaded: %d identities", len(self._gallery))
            return len(self._gallery)

    @property
    def gallery_size(self) -> int:
        return len(self._gallery)

    def _result(self, analyzed: AnalyzedFace) -> MatchResult:
        f = analyzed.face
        if analyzed.embedding is None:
            return MatchResult(MatchStatus.LOW_QUALITY, detection_score=f.score,
                               quality_reasons=analyzed.quality.reasons)
        with self._lock:
            d = self._matcher.decide(analyzed.embedding)
            name = self._names.get(d.person_id) if d.person_id else None
        if d.person_id is None:
            return MatchResult(MatchStatus.UNKNOWN, similarity=d.best_score,
                               runner_up_similarity=d.runner_up_score, detection_score=f.score)
        return MatchResult(MatchStatus.MATCH, person_id=d.person_id, full_name=name,
                           similarity=d.best_score, runner_up_similarity=d.runner_up_score,
                           detection_score=f.score)

    def recognize_faces(self, frame: np.ndarray, faces: list[DetectedFace]) -> list[FaceRecognition]:
        analyzed = self.engine.analyze(frame, faces, self.assessor)
        return [FaceRecognition(a.face, self._result(a)) for a in analyzed]

    def recognize_frame(self, frame: np.ndarray) -> list[FaceRecognition]:
        """Detect and recognise every face in a BGR frame."""
        return self.recognize_faces(frame, self.engine.detect(frame))
