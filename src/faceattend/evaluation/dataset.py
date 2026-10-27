"""Evaluation dataset layout and leakage checks.

<root>/<split>/                   split = val (threshold selection) or test (final report)
    gallery/<identity>/*.jpg      enrollment images
    probe/<identity>/*.jpg        genuine test images of enrolled identities (different captures!)
    impostor/<identity>/*.jpg     identities NOT in gallery (unknown-person rejection)
    conditions.csv                optional: relative_path,condition  (e.g. probe/s01/a.jpg,low_light)
    detection/                    optional: images for detection precision/recall
    detection_annotations.json    optional: {"img.jpg": [[x1,y1,x2,y2], ...]}
"""
from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

from ..exceptions import EvaluationDataError
from ..image_utils import IMAGE_EXTENSIONS


@dataclass
class EvalItem:
    path: Path
    rel: str
    identity: str


@dataclass
class EvalSplit:
    root: Path
    gallery: list[EvalItem]
    probe: list[EvalItem]
    impostor: list[EvalItem]
    conditions: dict[str, str] = field(default_factory=dict)
    detection_annotations: dict[str, list[list[float]]] | None = None
    warnings: list[str] = field(default_factory=list)


def _collect(root: Path, sub: str) -> list[EvalItem]:
    base = root / sub
    if not base.is_dir():
        return []
    items = []
    for ident in sorted(p for p in base.iterdir() if p.is_dir()):
        for f in sorted(ident.rglob("*")):
            if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS:
                items.append(EvalItem(f, f.relative_to(root).as_posix(), ident.name))
    return items


def file_hash(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def load_split(root: str | Path) -> EvalSplit:
    r = Path(root)
    if not r.is_dir():
        raise EvaluationDataError(f"Evaluation split not found: {r}")
    split = EvalSplit(r, _collect(r, "gallery"), _collect(r, "probe"), _collect(r, "impostor"))
    if not split.gallery:
        raise EvaluationDataError(f"{r}/gallery has no identity folders with images")
    cpath = r / "conditions.csv"
    if cpath.is_file():
        with cpath.open(newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                split.conditions[row["relative_path"].strip()] = row["condition"].strip()
    apath = r / "detection_annotations.json"
    if apath.is_file():
        split.detection_annotations = json.loads(apath.read_text(encoding="utf-8"))
    check_leakage(split)
    return split


def check_leakage(split: EvalSplit) -> None:
    """Raise on gallery/test overlap; record softer issues as warnings."""
    g_ids = {i.identity for i in split.gallery}
    imp_ids = {i.identity for i in split.impostor}
    overlap_ids = g_ids & imp_ids
    if overlap_ids:
        raise EvaluationDataError(f"Impostor identities also enrolled in gallery: {sorted(overlap_ids)}")
    g_hashes = {file_hash(i.path): i.rel for i in split.gallery}
    for item in split.probe + split.impostor:
        h = file_hash(item.path)
        if h in g_hashes:
            raise EvaluationDataError(f"Data leakage: {item.rel} is byte-identical to gallery image {g_hashes[h]}")
    missing = {i.identity for i in split.probe} - g_ids
    if missing:
        split.warnings.append(f"Probe identities without gallery images (counted as unknowns): "
                              f"{sorted(missing)}")
    if not split.impostor:
        split.warnings.append("No impostor identities: FAR is estimated only from cross-identity "
                              "probe comparisons; unknown-person rejection cannot be measured.")


def check_cross_split_leakage(a: EvalSplit, b: EvalSplit) -> list[str]:
    ha = {file_hash(i.path) for i in a.gallery + a.probe + a.impostor}
    dup = [i.rel for i in b.gallery + b.probe + b.impostor if file_hash(i.path) in ha]
    return [f"{len(dup)} images in {b.root.name} also appear in {a.root.name}"] if dup else []
