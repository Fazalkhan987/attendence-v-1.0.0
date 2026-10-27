"""ArcFace embedder using InsightFace's recognition model and its exact alignment."""
from __future__ import annotations

import warnings

import numpy as np

# scikit-image >= 0.26 deprecates an API used inside insightface.utils.face_align.
warnings.filterwarnings("ignore", category=FutureWarning, module=r"insightface\.utils\.face_align")

from ..exceptions import InvalidImageError
from ..schemas import DetectedFace
from .base import l2_normalize


class ArcFaceEmbedder:
    """Wraps a prepared ``insightface.model_zoo.arcface_onnx.ArcFaceONNX``.

    Alignment uses ``insightface.utils.face_align.norm_crop`` - the same 5-point
    similarity transform to the 112x112 ArcFace template the model was trained with.
    """

    def __init__(self, model, name: str):
        from insightface.utils import face_align  # imported lazily: optional at test time

        self._model = model
        self._norm_crop = face_align.norm_crop
        self.input_size = int(model.input_size[0])
        self.name = name
        self.embedding_dim = int(model.output_shape[1]) if getattr(model, "output_shape", None) else 512

    def align(self, image: np.ndarray, face: DetectedFace) -> np.ndarray:
        if face.landmarks is None or face.landmarks.shape != (5, 2):
            raise InvalidImageError("ArcFace alignment requires 5 facial landmarks")
        return self._norm_crop(image, landmark=face.landmarks, image_size=self.input_size)

    def embed(self, aligned: list[np.ndarray]) -> np.ndarray:
        if not aligned:
            return np.zeros((0, self.embedding_dim), dtype=np.float32)
        feats = self._model.get_feat(list(aligned))
        return l2_normalize(np.asarray(feats, dtype=np.float32).reshape(len(aligned), -1))
