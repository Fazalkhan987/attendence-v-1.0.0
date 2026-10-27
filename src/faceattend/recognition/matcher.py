"""Open-set matching of a probe embedding against the enrolled gallery (cosine similarity)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .base import l2_normalize


class EmbeddingGallery:
    """In-memory gallery of per-person embedding sets.

    Strategies (all cosine similarity on L2-normalised vectors):
      * ``centroid``  - similarity to the re-normalised mean of the person's samples (default;
                        robust to a single odd sample and O(persons) per probe)
      * ``max``       - best single-sample similarity (more permissive, more FAR risk)
      * ``topk_mean`` - mean of the k best sample similarities (in between)
    """

    def __init__(self, strategy: str = "centroid", top_k: int = 3):
        if strategy not in {"centroid", "max", "topk_mean"}:
            raise ValueError(f"Unknown strategy: {strategy}")
        self.strategy = strategy
        self.top_k = max(1, int(top_k))
        self._ids: list[str] = []
        self._samples = np.zeros((0, 0), dtype=np.float32)
        self._offsets = np.zeros(0, dtype=np.int64)
        self._counts = np.zeros(0, dtype=np.int64)
        self._centroids = np.zeros((0, 0), dtype=np.float32)

    def load(self, embeddings_by_person: dict[str, np.ndarray]) -> None:
        ids, blocks = [], []
        for pid in sorted(embeddings_by_person):
            emb = np.asarray(embeddings_by_person[pid], dtype=np.float32)
            if emb.ndim != 2 or emb.shape[0] == 0:
                continue
            ids.append(pid)
            blocks.append(l2_normalize(emb))
        self._ids = ids
        if not blocks:
            self._samples = np.zeros((0, 0), dtype=np.float32)
            self._centroids = np.zeros((0, 0), dtype=np.float32)
            self._offsets = np.zeros(0, dtype=np.int64)
            self._counts = np.zeros(0, dtype=np.int64)
            return
        dims = {b.shape[1] for b in blocks}
        if len(dims) != 1:
            raise ValueError(f"Inconsistent embedding dimensions in gallery: {dims}")
        self._counts = np.array([b.shape[0] for b in blocks], dtype=np.int64)
        self._offsets = np.concatenate([[0], np.cumsum(self._counts)[:-1]]).astype(np.int64)
        self._samples = np.vstack(blocks)
        self._centroids = l2_normalize(np.vstack([b.mean(axis=0) for b in blocks]))

    def __len__(self) -> int:
        return len(self._ids)

    @property
    def person_ids(self) -> list[str]:
        return list(self._ids)

    def centroid(self, person_id: str) -> np.ndarray | None:
        try:
            return self._centroids[self._ids.index(person_id)]
        except ValueError:
            return None

    def score_all(self, embedding: np.ndarray) -> np.ndarray:
        """Return one similarity per enrolled person (aligned with ``person_ids``)."""
        if not self._ids:
            return np.zeros(0, dtype=np.float32)
        q = l2_normalize(np.asarray(embedding, dtype=np.float32).reshape(-1))
        if self.strategy == "centroid":
            return self._centroids @ q
        sims = self._samples @ q
        if self.strategy == "max":
            return np.maximum.reduceat(sims, self._offsets)
        out = np.empty(len(self._ids), dtype=np.float32)
        for i, (o, c) in enumerate(zip(self._offsets, self._counts)):
            block = sims[o:o + c]
            k = min(self.top_k, c)
            out[i] = np.partition(block, c - k)[c - k:].mean()
        return out

    def ranked(self, embedding: np.ndarray, n: int = 2) -> list[tuple[str, float]]:
        scores = self.score_all(embedding)
        if scores.size == 0:
            return []
        order = np.argsort(-scores)[:n]
        return [(self._ids[i], float(scores[i])) for i in order]


@dataclass
class Decision:
    person_id: str | None
    best_score: float | None
    runner_up_score: float | None
    candidate_id: str | None     # top-1 identity even if rejected (for logging/eval)


class FaceMatcher:
    """Applies the open-set decision rule: accept top-1 only if it clears the threshold
    AND beats the runner-up identity by ``margin``; otherwise the face is Unknown."""

    def __init__(self, gallery: EmbeddingGallery, threshold: float, margin: float = 0.0):
        self.gallery = gallery
        self.threshold = float(threshold)
        self.margin = float(margin)

    def decide(self, embedding: np.ndarray) -> Decision:
        ranked = self.gallery.ranked(embedding, n=2)
        if not ranked:
            return Decision(None, None, None, None)
        best_id, best = ranked[0]
        runner = ranked[1][1] if len(ranked) > 1 else None
        accepted = best >= self.threshold and (runner is None or best - runner >= self.margin)
        return Decision(best_id if accepted else None, best, runner, best_id)
