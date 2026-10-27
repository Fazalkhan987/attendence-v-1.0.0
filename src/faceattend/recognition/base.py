"""Embedder interface: alignment + embedding, replaceable independently of detection."""
from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

from ..schemas import DetectedFace


@runtime_checkable
class FaceEmbedder(Protocol):
    name: str            # stored with every embedding; embeddings from different models never mix
    embedding_dim: int

    def align(self, image: np.ndarray, face: DetectedFace) -> np.ndarray:
        """Return the aligned crop the model expects (BGR uint8)."""
        ...

    def embed(self, aligned: list[np.ndarray]) -> np.ndarray:
        """Return (N, D) L2-normalised float32 embeddings for aligned crops."""
        ...


def l2_normalize(x: np.ndarray, axis: int = -1, eps: float = 1e-10) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return x / np.maximum(np.linalg.norm(x, axis=axis, keepdims=True), eps)
