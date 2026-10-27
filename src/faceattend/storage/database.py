"""SQLite repository for person records and face embeddings.

Embeddings are stored as raw little-endian float32 BLOBs (no pickle). Every embedding
row records the model that produced it, so embeddings from different models never mix.
No images are stored in the database.
"""
from __future__ import annotations

import json
import logging
import os
import sqlite3
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from ..exceptions import DuplicatePersonError, PersonNotFoundError

log = logging.getLogger(__name__)
SCHEMA_VERSION = 1

_SCHEMA = """
CREATE TABLE IF NOT EXISTS persons (
    person_id     TEXT PRIMARY KEY,
    full_name     TEXT NOT NULL,
    department    TEXT,
    class_name    TEXT,
    metadata_json TEXT,
    registered_at TEXT NOT NULL,
    updated_at    TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS face_samples (
    sample_id    INTEGER PRIMARY KEY AUTOINCREMENT,
    person_id    TEXT NOT NULL REFERENCES persons(person_id) ON DELETE CASCADE,
    model_name   TEXT NOT NULL,
    dim          INTEGER NOT NULL,
    embedding    BLOB NOT NULL,
    source       TEXT NOT NULL,
    quality_json TEXT,
    crop_path    TEXT,
    created_at   TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_samples_person ON face_samples(person_id);
CREATE INDEX IF NOT EXISTS idx_samples_model  ON face_samples(model_name);
CREATE TABLE IF NOT EXISTS settings (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS audit_log (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    ts        TEXT NOT NULL,
    action    TEXT NOT NULL,
    person_id TEXT,
    detail    TEXT
);
"""


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def encode_embedding(vec: np.ndarray) -> bytes:
    return np.asarray(vec, dtype="<f4").reshape(-1).tobytes()


def decode_embedding(blob: bytes, dim: int) -> np.ndarray:
    arr = np.frombuffer(blob, dtype="<f4")
    if arr.size != dim:
        raise ValueError(f"Corrupt embedding: expected {dim} values, got {arr.size}")
    return arr.astype(np.float32)


@dataclass
class Person:
    person_id: str
    full_name: str
    department: str | None
    class_name: str | None
    metadata: dict[str, Any]
    registered_at: str
    updated_at: str
    sample_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


@dataclass
class NewSample:
    embedding: np.ndarray
    source: str
    quality: dict[str, Any]
    crop_path: str | None = None


class Repository:
    """Thread-safe (single connection + lock) repository. Safe for the CLI and for a
    future FastAPI process; for multi-process deployments move to a server database."""

    def __init__(self, db_path: str | Path):
        self.path = Path(db_path)
        new_db = not self.path.exists()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        _restrict_permissions(self.path.parent, 0o700)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(str(self.path), check_same_thread=False, isolation_level=None)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._conn.execute("PRAGMA journal_mode = WAL")
        self._conn.execute("PRAGMA secure_delete = ON")   # overwrite deleted biometric rows
        self._conn.executescript(_SCHEMA)
        if new_db:
            _restrict_permissions(self.path, 0o600)
        version = self.get_setting("schema_version")
        if version is None:
            self.set_setting("schema_version", str(SCHEMA_VERSION))
        elif int(version) != SCHEMA_VERSION:
            raise RuntimeError(f"Unsupported DB schema version {version}")

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                yield self._conn
            except BaseException:
                self._conn.execute("ROLLBACK")
                raise
            else:
                self._conn.execute("COMMIT")

    # ---------- settings & audit ----------
    def get_setting(self, key: str) -> str | None:
        with self._lock:
            row = self._conn.execute("SELECT value FROM settings WHERE key=?", (key,)).fetchone()
        return None if row is None else row["value"]

    def set_setting(self, key: str, value: str) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO settings(key, value) VALUES(?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value", (key, value))

    def audit(self, action: str, person_id: str | None = None, detail: str | None = None,
              conn: sqlite3.Connection | None = None) -> None:
        """Audit entries never contain biometric data."""
        c = conn or self._conn
        with self._lock:
            c.execute("INSERT INTO audit_log(ts, action, person_id, detail) VALUES(?,?,?,?)",
                      (utcnow(), action, person_id, detail))

    # ---------- persons ----------
    def person_exists(self, person_id: str) -> bool:
        with self._lock:
            return self._conn.execute(
                "SELECT 1 FROM persons WHERE person_id=?", (person_id,)).fetchone() is not None

    def get_person(self, person_id: str) -> Person:
        with self._lock:
            row = self._conn.execute(
                "SELECT p.*, (SELECT COUNT(*) FROM face_samples s WHERE s.person_id=p.person_id) AS n "
                "FROM persons p WHERE p.person_id=?", (person_id,)).fetchone()
        if row is None:
            raise PersonNotFoundError(f"No person registered with ID '{person_id}'")
        return _row_to_person(row)

    def list_persons(self) -> list[Person]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT p.*, (SELECT COUNT(*) FROM face_samples s WHERE s.person_id=p.person_id) AS n "
                "FROM persons p ORDER BY p.person_id").fetchall()
        return [_row_to_person(r) for r in rows]

    def names(self) -> dict[str, str]:
        with self._lock:
            rows = self._conn.execute("SELECT person_id, full_name FROM persons").fetchall()
        return {r["person_id"]: r["full_name"] for r in rows}

    def create_person_with_samples(self, person_id: str, full_name: str, department: str | None,
                                   class_name: str | None, metadata: dict[str, Any],
                                   model_name: str, samples: list[NewSample]) -> None:
        now = utcnow()
        try:
            with self.transaction() as c:
                c.execute(
                    "INSERT INTO persons(person_id, full_name, department, class_name, metadata_json, "
                    "registered_at, updated_at) VALUES(?,?,?,?,?,?,?)",
                    (person_id, full_name, department, class_name, json.dumps(metadata), now, now))
                self._insert_samples(c, person_id, model_name, samples, now)
                self.audit("register", person_id, f"samples={len(samples)}", conn=c)
        except sqlite3.IntegrityError as exc:
            raise DuplicatePersonError(f"Person ID '{person_id}' is already registered") from exc

    def add_samples(self, person_id: str, model_name: str, samples: list[NewSample]) -> None:
        if not self.person_exists(person_id):
            raise PersonNotFoundError(f"No person registered with ID '{person_id}'")
        now = utcnow()
        with self.transaction() as c:
            self._insert_samples(c, person_id, model_name, samples, now)
            c.execute("UPDATE persons SET updated_at=? WHERE person_id=?", (now, person_id))
            self.audit("add_samples", person_id, f"samples={len(samples)}", conn=c)

    def update_person(self, person_id: str, **fields: Any) -> Person:
        allowed = {"full_name", "department", "class_name", "metadata"}
        bad = set(fields) - allowed
        if bad:
            raise ValueError(f"Cannot update fields: {sorted(bad)}")
        if not self.person_exists(person_id):
            raise PersonNotFoundError(f"No person registered with ID '{person_id}'")
        sets, vals = [], []
        for k, v in fields.items():
            if k == "metadata":
                sets.append("metadata_json=?")
                vals.append(json.dumps(v))
            else:
                sets.append(f"{k}=?")
                vals.append(v)
        if sets:
            sets.append("updated_at=?")
            vals.extend([utcnow(), person_id])
            with self.transaction() as c:
                c.execute(f"UPDATE persons SET {', '.join(sets)} WHERE person_id=?", vals)
                self.audit("update", person_id, ",".join(sorted(fields)), conn=c)
        return self.get_person(person_id)

    def delete_person(self, person_id: str) -> list[str]:
        """Delete person + all embeddings. Returns crop paths the caller must delete from disk."""
        with self.transaction() as c:
            if c.execute("SELECT 1 FROM persons WHERE person_id=?", (person_id,)).fetchone() is None:
                raise PersonNotFoundError(f"No person registered with ID '{person_id}'")
            crops = [r["crop_path"] for r in c.execute(
                "SELECT crop_path FROM face_samples WHERE person_id=? AND crop_path IS NOT NULL",
                (person_id,))]
            c.execute("DELETE FROM face_samples WHERE person_id=?", (person_id,))
            c.execute("DELETE FROM persons WHERE person_id=?", (person_id,))
            self.audit("delete", person_id, None, conn=c)
        with self._lock:
            self._conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        return crops

    # ---------- embeddings ----------
    def sample_count(self, person_id: str) -> int:
        with self._lock:
            return int(self._conn.execute(
                "SELECT COUNT(*) FROM face_samples WHERE person_id=?", (person_id,)).fetchone()[0])

    def load_embeddings(self, model_name: str) -> dict[str, np.ndarray]:
        out: dict[str, list[np.ndarray]] = {}
        with self._lock:
            rows = self._conn.execute(
                "SELECT person_id, dim, embedding FROM face_samples WHERE model_name=? ORDER BY sample_id",
                (model_name,)).fetchall()
            other = self._conn.execute(
                "SELECT COUNT(*) FROM face_samples WHERE model_name<>?", (model_name,)).fetchone()[0]
        if other:
            log.warning("%d stored embeddings belong to a different model and are ignored; "
                        "re-enroll those persons for model '%s'", other, model_name)
        for r in rows:
            out.setdefault(r["person_id"], []).append(decode_embedding(r["embedding"], r["dim"]))
        return {k: np.vstack(v) for k, v in out.items()}

    @staticmethod
    def _insert_samples(c: sqlite3.Connection, person_id: str, model_name: str,
                        samples: list[NewSample], now: str) -> None:
        for s in samples:
            emb = np.asarray(s.embedding, dtype=np.float32).reshape(-1)
            c.execute(
                "INSERT INTO face_samples(person_id, model_name, dim, embedding, source, quality_json, "
                "crop_path, created_at) VALUES(?,?,?,?,?,?,?,?)",
                (person_id, model_name, emb.size, encode_embedding(emb), s.source,
                 json.dumps(s.quality), s.crop_path, now))


def _row_to_person(row: sqlite3.Row) -> Person:
    return Person(
        person_id=row["person_id"], full_name=row["full_name"], department=row["department"],
        class_name=row["class_name"], metadata=json.loads(row["metadata_json"] or "{}"),
        registered_at=row["registered_at"], updated_at=row["updated_at"], sample_count=int(row["n"]))


def _restrict_permissions(path: Path, mode: int) -> None:
    if os.name == "posix":
        try:
            os.chmod(path, mode)
        except OSError:
            log.warning("Could not restrict permissions on %s", path)
