"""Logging configuration. Never log embeddings, images or other raw biometric data."""
from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

from .config import LoggingConfig

_FMT = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"


def setup_logging(cfg: LoggingConfig) -> None:
    root = logging.getLogger()
    root.setLevel(cfg.level.upper())
    for h in list(root.handlers):
        root.removeHandler(h)
    console = logging.StreamHandler()
    console.setFormatter(logging.Formatter(_FMT))
    root.addHandler(console)
    if cfg.file:
        path = Path(cfg.file)
        path.parent.mkdir(parents=True, exist_ok=True)
        fh = RotatingFileHandler(path, maxBytes=5_000_000, backupCount=3, encoding="utf-8")
        fh.setFormatter(logging.Formatter(_FMT))
        root.addHandler(fh)
    # Third-party libraries are chatty at INFO.
    for noisy in ("onnxruntime", "matplotlib", "PIL"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
