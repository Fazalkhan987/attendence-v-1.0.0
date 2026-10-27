"""Pure metric functions (no model dependencies) - unit tested."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def verification_rates(genuine: np.ndarray, impostor: np.ndarray, threshold: float) -> dict[str, float]:
    g, i = np.asarray(genuine, float), np.asarray(impostor, float)
    tar = float((g >= threshold).mean()) if g.size else float("nan")
    far = float((i >= threshold).mean()) if i.size else float("nan")
    return {"threshold": float(threshold), "TAR": tar, "FRR": 1.0 - tar if g.size else float("nan"),
            "FAR": far}


def threshold_at_far(impostor: np.ndarray, target_far: float) -> float:
    """Smallest threshold whose empirical FAR (score >= thr) is <= target_far."""
    imp = np.sort(np.asarray(impostor, float))[::-1]
    if imp.size == 0:
        raise ValueError("No impostor scores")
    allowed = int(np.floor(target_far * imp.size))
    if allowed >= imp.size:
        return float(np.nextafter(imp[-1], -np.inf))
    return float(np.nextafter(imp[allowed], np.inf))


def equal_error_rate(genuine: np.ndarray, impostor: np.ndarray) -> tuple[float, float]:
    """Return (EER, threshold) by sweeping all observed scores."""
    g, i = np.asarray(genuine, float), np.asarray(impostor, float)
    thresholds = np.unique(np.concatenate([g, i]))
    best = (1.0, float(thresholds[0]) if thresholds.size else 0.0, 1.0)
    for t in thresholds:
        frr = float((g < t).mean())
        far = float((i >= t).mean())
        gap = abs(frr - far)
        if gap < best[2]:
            best = ((frr + far) / 2.0, float(t), gap)
    return best[0], best[1]


def sweep(genuine: np.ndarray, impostor: np.ndarray, thresholds: np.ndarray) -> list[dict[str, float]]:
    return [verification_rates(genuine, impostor, float(t)) for t in thresholds]


@dataclass
class OpenSetOutcome:
    true_id: str | None          # None for an unenrolled (unknown) probe
    predicted_id: str | None     # None if rejected
    top1_id: str | None          # rank-1 identity regardless of threshold


def open_set_metrics(outcomes: list[OpenSetOutcome]) -> dict[str, float | int]:
    known = [o for o in outcomes if o.true_id is not None]
    unknown = [o for o in outcomes if o.true_id is None]
    correct = sum(o.predicted_id == o.true_id for o in known)
    misid = sum(o.predicted_id is not None and o.predicted_id != o.true_id for o in known)
    rank1 = sum(o.top1_id == o.true_id for o in known)
    unk_accepted = sum(o.predicted_id is not None for o in unknown)
    nan = float("nan")
    return {
        "known_probes": len(known),
        "unknown_probes": len(unknown),
        "identification_accuracy": correct / len(known) if known else nan,     # DIR at threshold
        "misidentification_rate": misid / len(known) if known else nan,
        "false_rejection_rate_known": (len(known) - correct - misid) / len(known) if known else nan,
        "closed_set_rank1_accuracy": rank1 / len(known) if known else nan,
        "unknown_rejection_rate": (len(unknown) - unk_accepted) / len(unknown) if unknown else nan,
        "false_positive_identification_rate": unk_accepted / len(unknown) if unknown else nan,
    }


def box_iou(a, b) -> float:
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def detection_counts(pred: list[tuple[list[float], float]], gt: list[list[float]],
                     iou_threshold: float = 0.5) -> tuple[int, int, int]:
    """Greedy matching by score. Returns (TP, FP, FN) for one image."""
    matched: set[int] = set()
    tp = fp = 0
    for box, _score in sorted(pred, key=lambda p: -p[1]):
        best_j, best_iou = -1, iou_threshold
        for j, g in enumerate(gt):
            if j in matched:
                continue
            v = box_iou(box, g)
            if v >= best_iou:
                best_j, best_iou = j, v
        if best_j >= 0:
            matched.add(best_j)
            tp += 1
        else:
            fp += 1
    return tp, fp, len(gt) - len(matched)


def latency_stats(ms: list[float]) -> dict[str, float]:
    if not ms:
        return {}
    a = np.asarray(ms, float)
    return {"n": int(a.size), "mean_ms": float(a.mean()), "p50_ms": float(np.percentile(a, 50)),
            "p95_ms": float(np.percentile(a, 95)), "max_ms": float(a.max())}
