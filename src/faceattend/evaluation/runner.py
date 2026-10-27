"""End-to-end evaluation using the real engine. Produces measured numbers only."""
from __future__ import annotations

import logging
import platform
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..config import AppConfig
from ..exceptions import InvalidImageError
from ..image_utils import read_image
from ..quality import QualityAssessor
from ..recognition.matcher import EmbeddingGallery, FaceMatcher
from ..services.engine import FaceEngine
from .dataset import EvalItem, EvalSplit
from .metrics import (OpenSetOutcome, detection_counts, equal_error_rate, latency_stats,
                      open_set_metrics, sweep, threshold_at_far, verification_rates)

log = logging.getLogger(__name__)


@dataclass(eq=False)
class _Embedded:
    item: EvalItem
    embedding: np.ndarray | None
    failure: str | None


@dataclass
class Timings:
    detect_ms: list[float] = field(default_factory=list)
    embed_ms: list[float] = field(default_factory=list)


def _embed_items(engine: FaceEngine, items: list[EvalItem], assessor: QualityAssessor,
                 timings: Timings) -> list[_Embedded]:
    out = []
    for it in items:
        try:
            img = read_image(it.path)
        except InvalidImageError as exc:
            out.append(_Embedded(it, None, f"unreadable: {exc}"))
            continue
        t0 = time.perf_counter()
        faces = engine.detect(img)
        timings.detect_ms.append((time.perf_counter() - t0) * 1000)
        if not faces:
            out.append(_Embedded(it, None, "no face detected"))
            continue
        face = max(faces, key=lambda f: f.size)   # evaluation images: use the largest face
        t1 = time.perf_counter()
        a = engine.analyze(img, [face], assessor)[0]
        timings.embed_ms.append((time.perf_counter() - t1) * 1000)
        if a.embedding is None:
            out.append(_Embedded(it, None, "quality: " + "; ".join(a.quality.reasons)))
        else:
            out.append(_Embedded(it, a.embedding, None))
    return out


def _hardware() -> dict[str, Any]:
    info = {"platform": platform.platform(), "processor": platform.processor() or platform.machine(),
            "python": platform.python_version()}
    try:
        import onnxruntime as ort
        info["onnxruntime"] = ort.__version__
        info["available_providers"] = ort.get_available_providers()
    except Exception:
        pass
    return info


def evaluate_split(engine: FaceEngine, cfg: AppConfig, split: EvalSplit,
                   threshold: float | None = None, target_far: float = 1e-3) -> dict[str, Any]:
    timings = Timings()
    enroll_q = QualityAssessor(cfg.enrollment_quality)
    recog_q = QualityAssessor(cfg.recognition_quality)

    # 1) Build an isolated gallery (never touches the production database).
    g_emb = _embed_items(engine, split.gallery, enroll_q, timings)
    by_person: dict[str, list[np.ndarray]] = {}
    for e in g_emb:
        if e.embedding is not None:
            by_person.setdefault(e.item.identity, []).append(e.embedding)
    gallery = EmbeddingGallery(cfg.recognition.match_strategy, cfg.recognition.top_k)
    gallery.load({k: np.vstack(v) for k, v in by_person.items()})
    ids = gallery.person_ids

    # 2) Probe + impostor embeddings.
    p_emb = _embed_items(engine, split.probe, recog_q, timings)
    i_emb = _embed_items(engine, split.impostor, recog_q, timings)

    # Unenrolled identities (impostors, or probes with no gallery images) have true_id None.
    tested = [(e, e.item.identity if e.item.identity in ids else None) for e in p_emb] + \
             [(e, None) for e in i_emb]
    genuine, impostor = [], []
    for e, true in tested:
        if e.embedding is None:
            continue
        scores = gallery.score_all(e.embedding)
        for pid, s in zip(ids, scores):
            (genuine if pid == true else impostor).append(float(s))
    genuine_a, impostor_a = np.array(genuine), np.array(impostor)

    report: dict[str, Any] = {
        "split": str(split.root), "model": engine.model_name, "hardware": _hardware(),
        "config": {"match_strategy": cfg.recognition.match_strategy, "margin": cfg.recognition.margin,
                   "detection_threshold": cfg.detection.confidence_threshold},
        "dataset": {
            "gallery_identities": len(ids), "gallery_images": len(split.gallery),
            "gallery_images_used": sum(e.embedding is not None for e in g_emb),
            "probe_images": len(split.probe), "impostor_images": len(split.impostor),
            "impostor_identities": len({i.identity for i in split.impostor}),
            "genuine_comparisons": len(genuine), "impostor_comparisons": len(impostor),
        },
        "warnings": list(split.warnings),
    }
    fails = [e for e in p_emb + i_emb if e.failure]
    report["failure_to_acquire"] = {
        "count": len(fails), "rate": len(fails) / max(1, len(p_emb) + len(i_emb)),
        "examples": [f"{e.item.rel}: {e.failure}" for e in fails[:20]],
    }
    if not genuine or not impostor:
        report["warnings"].append("Not enough genuine/impostor comparisons to compute verification metrics.")
        report["latency"] = {"detection": latency_stats(timings.detect_ms),
                             "align_quality_embed": latency_stats(timings.embed_ms)}
        return report

    min_far = 1.0 / len(impostor)
    if target_far < min_far:
        report["warnings"].append(
            f"target FAR {target_far:g} is below the resolution of this dataset (1/{len(impostor)} = "
            f"{min_far:.2e}); the chosen threshold is not statistically supported. Add impostor data.")
    thr_far = threshold_at_far(impostor_a, target_far)
    eer, eer_thr = equal_error_rate(genuine_a, impostor_a)
    if threshold is not None:
        used_thr = float(threshold)
    elif target_far >= min_far:
        used_thr = thr_far
    else:
        used_thr = cfg.recognition.similarity_threshold
        report["warnings"].append(f"Identification metrics use the configured threshold {used_thr} "
                                  "because the target-FAR threshold is not supported by the data.")
    report["verification"] = {
        "genuine_score": {"mean": float(genuine_a.mean()), "min": float(genuine_a.min())},
        "impostor_score": {"mean": float(impostor_a.mean()), "max": float(impostor_a.max())},
        "EER": eer, "EER_threshold": eer_thr,
        "target_FAR": target_far, "threshold_at_target_FAR": thr_far,
        "at_target_FAR": verification_rates(genuine_a, impostor_a, thr_far),
        "at_used_threshold": verification_rates(genuine_a, impostor_a, used_thr),
        "sweep": sweep(genuine_a, impostor_a, np.round(np.arange(0.20, 0.71, 0.05), 2)),
    }

    # 3) Open-set identification at the used threshold (+ margin), plus FTA counted as rejections.
    matcher = FaceMatcher(gallery, used_thr, cfg.recognition.margin)
    outcomes, per_condition = [], {}
    for e, true in tested:
        if e.embedding is None:
            o = OpenSetOutcome(true, None, None)
        else:
            d = matcher.decide(e.embedding)
            o = OpenSetOutcome(true, d.person_id, d.candidate_id)
        outcomes.append(o)
        cond = split.conditions.get(e.item.rel)
        if cond:
            per_condition.setdefault(cond, []).append(o)
    report["identification"] = {"threshold": used_thr, **open_set_metrics(outcomes)}
    if per_condition:
        report["by_condition"] = {c: open_set_metrics(v) for c, v in sorted(per_condition.items())}

    # 4) Detection precision/recall if annotations provided.
    if split.detection_annotations:
        tp = fp = fn = 0
        for rel, gt in split.detection_annotations.items():
            p = split.root / "detection" / rel
            try:
                img = read_image(p)
            except InvalidImageError as exc:
                report["warnings"].append(f"detection image skipped: {exc}")
                continue
            preds = [(f.bbox.tolist(), f.score) for f in engine.detect(img)]
            a, b, c = detection_counts(preds, gt)
            tp, fp, fn = tp + a, fp + b, fn + c
        report["detection"] = {
            "iou_threshold": 0.5, "TP": tp, "FP": fp, "FN": fn,
            "precision": tp / (tp + fp) if tp + fp else float("nan"),
            "recall": tp / (tp + fn) if tp + fn else float("nan"),
        }
    report["latency"] = {"detection": latency_stats(timings.detect_ms),
                         "align_quality_embed": latency_stats(timings.embed_ms)}
    return report
