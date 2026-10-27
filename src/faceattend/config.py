"""Typed configuration loaded from YAML with strict key checking."""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .exceptions import ConfigError


@dataclass
class ModelConfig:
    pack: str = "buffalo_l"
    root: str = "~/.insightface"
    providers: list[str] = field(default_factory=lambda: ["CPUExecutionProvider"])
    ctx_id: int = -1
    det_size: tuple[int, int] = (640, 640)


@dataclass
class DetectionConfig:
    confidence_threshold: float = 0.5
    nms_threshold: float = 0.4
    max_faces: int = 0


@dataclass
class RecognitionConfig:
    similarity_threshold: float = 0.45
    margin: float = 0.05
    match_strategy: str = "centroid"
    top_k: int = 3


@dataclass
class QualityConfig:
    min_face_size: int = 80
    min_det_score: float = 0.70
    min_blur_variance: float = 60.0
    min_brightness: float = 50.0
    max_brightness: float = 210.0
    min_contrast: float = 25.0
    max_yaw_deviation: float = 0.15
    max_roll_degrees: float = 15.0
    pitch_range: tuple[float, float] = (0.30, 0.90)


def _recognition_quality() -> QualityConfig:
    return QualityConfig(
        min_face_size=48, min_det_score=0.55, min_blur_variance=25.0,
        min_brightness=35.0, max_brightness=225.0, min_contrast=15.0,
        max_yaw_deviation=0.25, max_roll_degrees=35.0, pitch_range=(0.25, 0.95),
    )


@dataclass
class EnrollmentConfig:
    min_samples: int = 5
    target_samples: int = 8
    max_samples_per_person: int = 30
    duplicate_identity_threshold: float = 0.55
    consistency_threshold: float = 0.35
    near_duplicate_threshold: float = 0.97
    auto_capture_interval_s: float = 0.7
    store_aligned_crops: bool = False


@dataclass
class LiveConfig:
    camera_index: int = 0
    backend: str = "auto"
    width: int = 1280
    height: int = 720
    recognize_every_n_frames: int = 5
    track_iou_threshold: float = 0.3
    track_max_missed: int = 10
    smoothing_window: int = 7
    min_votes: int = 4
    max_consecutive_read_failures: int = 30


@dataclass
class StorageConfig:
    db_path: str = "data/faceattend.db"
    crops_dir: str = "data/crops"
    reports_dir: str = "reports"


@dataclass
class SecurityConfig:
    require_admin_auth: bool = True


@dataclass
class LoggingConfig:
    level: str = "INFO"
    file: str | None = "logs/faceattend.log"


@dataclass
class AppConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    recognition: RecognitionConfig = field(default_factory=RecognitionConfig)
    enrollment_quality: QualityConfig = field(default_factory=QualityConfig)
    recognition_quality: QualityConfig = field(default_factory=_recognition_quality)
    enrollment: EnrollmentConfig = field(default_factory=EnrollmentConfig)
    live: LiveConfig = field(default_factory=LiveConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    def validate(self) -> None:
        r = self.recognition
        if not -1.0 <= r.similarity_threshold <= 1.0:
            raise ConfigError("recognition.similarity_threshold must be in [-1, 1]")
        if r.match_strategy not in {"centroid", "max", "topk_mean"}:
            raise ConfigError("recognition.match_strategy must be centroid|max|topk_mean")
        if r.margin < 0:
            raise ConfigError("recognition.margin must be >= 0")
        e = self.enrollment
        if not 1 <= e.min_samples <= e.target_samples <= e.max_samples_per_person:
            raise ConfigError("enrollment: need 1 <= min_samples <= target_samples <= max_samples_per_person")
        lv = self.live
        if lv.min_votes > lv.smoothing_window:
            raise ConfigError("live.min_votes cannot exceed live.smoothing_window")
        if not 0 < self.detection.confidence_threshold < 1:
            raise ConfigError("detection.confidence_threshold must be in (0, 1)")


def _build(cls, data: dict[str, Any], path: str):
    if not isinstance(data, dict):
        raise ConfigError(f"'{path}' must be a mapping")
    known = {f.name: f for f in dataclasses.fields(cls)}
    unknown = set(data) - set(known)
    if unknown:
        raise ConfigError(f"Unknown config keys under '{path}': {sorted(unknown)}")
    default = cls()
    kwargs = {}
    for name, f in known.items():
        if name not in data:
            continue
        current = getattr(default, name)
        value = data[name]
        if dataclasses.is_dataclass(current):
            kwargs[name] = _build(type(current), value, f"{path}.{name}")
        elif isinstance(current, tuple):
            kwargs[name] = tuple(value)
        else:
            kwargs[name] = value
    return cls(**kwargs)


def load_config(path: str | Path | None = None) -> AppConfig:
    """Load config from YAML. ``None`` → ``config/default.yaml`` if present, else built-in defaults."""
    if path is None:
        candidate = Path("config/default.yaml")
        path = candidate if candidate.exists() else None
    if path is None:
        cfg = AppConfig()
    else:
        p = Path(path)
        if not p.is_file():
            raise ConfigError(f"Config file not found: {p}")
        try:
            raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        except yaml.YAMLError as exc:
            raise ConfigError(f"Invalid YAML in {p}: {exc}") from exc
        cfg = _build(AppConfig, raw, "root")
    cfg.validate()
    return cfg
