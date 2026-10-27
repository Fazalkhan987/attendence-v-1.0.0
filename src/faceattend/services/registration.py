"""Person registration: sample screening, consistency/duplicate checks, CRUD."""
from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from ..config import EnrollmentConfig
from ..exceptions import (DuplicateIdentityError, DuplicatePersonError, EnrollmentError,
                          InvalidImageError)
from ..image_utils import list_images, read_image, write_image
from ..quality import QualityAssessor
from ..recognition.base import l2_normalize
from ..recognition.matcher import EmbeddingGallery
from ..storage.database import NewSample, Person, Repository
from ..validation import clean_optional, validate_name, validate_person_id
from .engine import FaceEngine

log = logging.getLogger(__name__)


@dataclass(eq=False)
class EnrollmentSample:
    embedding: np.ndarray
    aligned: np.ndarray
    quality: dict[str, Any]
    source: str


@dataclass
class Rejection:
    source: str
    reason: str


@dataclass
class ScreeningResult:
    accepted: list[EnrollmentSample] = field(default_factory=list)
    rejected: list[Rejection] = field(default_factory=list)


@dataclass
class RegistrationResult:
    person: Person
    stored_samples: int
    dropped: list[Rejection]


class RegistrationService:
    def __init__(self, engine: FaceEngine, repo: Repository, cfg: EnrollmentConfig,
                 assessor: QualityAssessor, crops_dir: str | Path):
        self.engine = engine
        self.repo = repo
        self.cfg = cfg
        self.assessor = assessor
        self.crops_dir = Path(crops_dir)

    # ---------- screening ----------
    def screen_image(self, image: np.ndarray, source: str) -> EnrollmentSample | Rejection:
        """Validate a single enrollment image: exactly one face, passing quality, embedded."""
        try:
            faces = self.engine.detect(image)
        except InvalidImageError as exc:
            return Rejection(source, str(exc))
        if not faces:
            return Rejection(source, "no face detected")
        if len(faces) > 1:
            return Rejection(source, f"{len(faces)} faces detected - enrollment needs exactly one")
        a = self.engine.analyze(image, faces, self.assessor)[0]
        if not a.quality.passed:
            return Rejection(source, "; ".join(a.quality.reasons))
        if a.embedding is None or a.aligned is None:
            return Rejection(source, "could not compute embedding")
        return EnrollmentSample(a.embedding, a.aligned, a.quality.metrics, source)

    def screen_directory(self, directory: str | Path) -> ScreeningResult:
        res = ScreeningResult()
        paths = list_images(directory)
        if not paths:
            raise EnrollmentError(f"No images (.jpg/.png/.bmp/.webp) found in {directory}")
        for p in paths:
            try:
                img = read_image(p)
            except InvalidImageError as exc:
                res.rejected.append(Rejection(p.name, str(exc)))
                continue
            out = self.screen_image(img, f"file:{p.name}")
            (res.accepted if isinstance(out, EnrollmentSample) else res.rejected).append(out)
        return res

    # ---------- sample set checks ----------
    def _filter_samples(self, samples: list[EnrollmentSample],
                        reference: np.ndarray | None) -> tuple[list[EnrollmentSample], list[Rejection]]:
        """Drop near-identical frames and samples inconsistent with the identity."""
        kept: list[EnrollmentSample] = []
        dropped: list[Rejection] = []
        for s in samples:
            if kept:
                sims = np.vstack([k.embedding for k in kept]) @ s.embedding
                if float(sims.max()) >= self.cfg.near_duplicate_threshold:
                    dropped.append(Rejection(s.source, "near-identical to another sample"))
                    continue
            kept.append(s)
        ref = reference
        if ref is None and len(kept) >= 2:
            # Robust reference: medoid (sample with the highest mean similarity to the others).
            E = np.vstack([k.embedding for k in kept])
            ref = E[int(np.argmax((E @ E.T).mean(axis=1)))]
        if ref is not None:
            consistent = []
            for s in kept:
                sim = float(s.embedding @ ref)
                if sim < self.cfg.consistency_threshold:
                    dropped.append(Rejection(s.source, f"inconsistent with identity (sim {sim:.2f}) "
                                                       "- possibly a different person"))
                else:
                    consistent.append(s)
            kept = consistent
        return kept, dropped

    def find_duplicate_identity(self, embeddings: np.ndarray,
                                exclude: str | None = None) -> tuple[str, float] | None:
        gallery = EmbeddingGallery("centroid")
        data = self.repo.load_embeddings(self.engine.model_name)
        if exclude:
            data.pop(exclude, None)
        gallery.load(data)
        if not len(gallery):
            return None
        probe = l2_normalize(np.asarray(embeddings).mean(axis=0))
        best = gallery.ranked(probe, n=1)[0]
        return best if best[1] >= self.cfg.duplicate_identity_threshold else None

    def _to_new_samples(self, person_id: str, samples: list[EnrollmentSample]) -> list[NewSample]:
        out = []
        for s in samples:
            crop_path = None
            if self.cfg.store_aligned_crops:
                p = self.crops_dir / person_id / f"{uuid.uuid4().hex}.png"
                write_image(p, s.aligned)
                crop_path = str(p)
            out.append(NewSample(s.embedding, s.source, s.quality, crop_path))
        return out

    # ---------- public operations ----------
    def register(self, person_id: str, full_name: str, samples: list[EnrollmentSample],
                 department: str | None = None, class_name: str | None = None,
                 metadata: dict[str, Any] | None = None, force: bool = False) -> RegistrationResult:
        pid = validate_person_id(person_id)
        name = validate_name(full_name)
        dept = clean_optional(department, "department")
        cls = clean_optional(class_name, "class")
        if self.repo.person_exists(pid):
            raise DuplicatePersonError(f"Person ID '{pid}' is already registered "
                                       "(use add-samples or update instead)")
        kept, dropped = self._filter_samples(samples, reference=None)
        kept = kept[: self.cfg.max_samples_per_person]
        if len(kept) < self.cfg.min_samples:
            raise EnrollmentError(f"Only {len(kept)} usable face samples; at least "
                                  f"{self.cfg.min_samples} required. Nothing was saved."
                                  + _drop_summary(dropped))
        dup = self.find_duplicate_identity(np.vstack([s.embedding for s in kept]))
        if dup and not force:
            raise DuplicateIdentityError(*dup)
        if dup:
            log.warning("Registering %s despite similarity %.3f to %s (forced)", pid, dup[1], dup[0])
        new = self._to_new_samples(pid, kept)
        try:
            self.repo.create_person_with_samples(pid, name, dept, cls, metadata or {},
                                                 self.engine.model_name, new)
        except Exception:
            self._remove_crops([s.crop_path for s in new if s.crop_path])
            raise
        log.info("Registered %s with %d samples (%d dropped)", pid, len(kept), len(dropped))
        return RegistrationResult(self.repo.get_person(pid), len(kept), dropped)

    def add_samples(self, person_id: str, samples: list[EnrollmentSample]) -> RegistrationResult:
        pid = validate_person_id(person_id)
        person = self.repo.get_person(pid)
        existing = self.repo.load_embeddings(self.engine.model_name).get(pid)
        reference = None if existing is None else l2_normalize(existing.mean(axis=0))
        kept, dropped = self._filter_samples(samples, reference)
        room = self.cfg.max_samples_per_person - person.sample_count
        if room <= 0:
            raise EnrollmentError(f"{pid} already has the maximum {self.cfg.max_samples_per_person} samples")
        kept = kept[:room]
        if not kept:
            raise EnrollmentError("No usable new samples. Nothing was saved.")
        if reference is None and len(kept) < self.cfg.min_samples:
            raise EnrollmentError(f"No samples exist for the current model; need at least "
                                  f"{self.cfg.min_samples} new samples.")
        new = self._to_new_samples(pid, kept)
        try:
            self.repo.add_samples(pid, self.engine.model_name, new)
        except Exception:
            self._remove_crops([s.crop_path for s in new if s.crop_path])
            raise
        return RegistrationResult(self.repo.get_person(pid), len(kept), dropped)

    @staticmethod
    def _remove_crops(paths: list[str]) -> None:
        remove_files(paths)


def _drop_summary(dropped: list[Rejection]) -> str:
    if not dropped:
        return ""
    near = sum("near-identical" in d.reason for d in dropped)
    inconsistent = len(dropped) - near
    return (f" Dropped {near} near-identical and {inconsistent} inconsistent sample(s) - "
            "capture more varied images (angle, expression, distance, lighting).")


def remove_files(paths: list[str]) -> None:
    for p in paths:
        try:
            Path(p).unlink(missing_ok=True)
        except OSError:
            log.warning("Could not delete crop file %s", p)
