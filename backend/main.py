"""FastAPI adapter for the FaceAttend engine.

The model and biometric repository remain owned by the faceattend package. This
module only translates HTTP requests into service calls and stores attendance
events in the same SQLite database.
"""
from __future__ import annotations

import json
import logging
import os
import sqlite3
import threading
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, Request, UploadFile, WebSocket, WebSocketDisconnect, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from faceattend.config import AppConfig, load_config
from faceattend.exceptions import (
    AuthError,
    DuplicateIdentityError,
    DuplicatePersonError,
    EnrollmentError,
    InvalidImageError,
    PersonNotFoundError,
    ValidationError,
)
from faceattend.image_utils import validate_frame
from faceattend.quality import QualityAssessor
from faceattend.security import AdminGuard
from faceattend.services.engine import FaceEngine, build_engine
from faceattend.services.recognition import RecognitionService
from faceattend.services.registration import EnrollmentSample, RegistrationService
from faceattend.storage.database import Person, Repository
from faceattend.camera.tracker import FaceTracker, TrackState
from faceattend.camera.overlay import draw_status, draw_track
from faceattend.validation import validate_name, validate_person_id

log = logging.getLogger(__name__)


class AttendanceCreate(BaseModel):
    person_id: str
    event_type: str = Field(default="check-in", pattern="^(check-in|check-out)$")
    note: str | None = Field(default=None, max_length=500)


class LoginRequest(BaseModel):
    username: str
    password: str


class AppState:
    def __init__(self, cfg: AppConfig):
        self.cfg = cfg
        self.repo = Repository(cfg.storage.db_path)
        self.engine: FaceEngine | None = None
        self.registration: RegistrationService | None = None
        self.recognition: RecognitionService | None = None
        self.admin = AdminGuard(self.repo, enabled=cfg.security.require_admin_auth)
        self.attendance = AttendanceRepository(cfg.storage.db_path)
        self.live_trackers: dict[str, tuple[FaceTracker, float]] = {}
        self.live_trackers_lock = threading.RLock()

    def load_model(self) -> None:
        self.engine = build_engine(self.cfg)
        self.registration = RegistrationService(
            self.engine,
            self.repo,
            self.cfg.enrollment,
            QualityAssessor(self.cfg.enrollment_quality),
            self.cfg.storage.crops_dir,
        )
        self.recognition = RecognitionService(
            self.engine,
            self.repo,
            self.cfg.recognition,
            QualityAssessor(self.cfg.recognition_quality),
        )

    def close(self) -> None:
        self.attendance.close()
        self.repo.close()


class AttendanceRepository:
    """Attendance persistence kept beside the package's biometric tables."""

    def __init__(self, db_path: str | Path):
        self.path = Path(db_path)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS attendance_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                person_id TEXT NOT NULL REFERENCES persons(person_id) ON DELETE CASCADE,
                event_type TEXT NOT NULL CHECK(event_type IN ('check-in', 'check-out')),
                note TEXT,
                created_at TEXT NOT NULL
            )
            """
        )
        self._conn.commit()

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def create(self, event: AttendanceCreate) -> dict[str, Any]:
        created_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
        with self._lock:
            cursor = self._conn.execute(
                "INSERT INTO attendance_events(person_id, event_type, note, created_at) "
                "VALUES (?, ?, ?, ?)",
                (event.person_id, event.event_type, event.note, created_at),
            )
            self._conn.commit()
        return {
            "id": cursor.lastrowid,
            "person_id": event.person_id,
            "event_type": event.event_type,
            "note": event.note,
            "created_at": created_at,
        }

    def list(self, person_id: str | None = None, limit: int = 100) -> list[dict[str, Any]]:
        with self._lock:
            if person_id:
                rows = self._conn.execute(
                    "SELECT id, person_id, event_type, note, created_at FROM attendance_events "
                    "WHERE person_id=? ORDER BY id DESC LIMIT ?",
                    (person_id, limit),
                ).fetchall()
            else:
                rows = self._conn.execute(
                    "SELECT id, person_id, event_type, note, created_at FROM attendance_events "
                    "ORDER BY id DESC LIMIT ?",
                    (limit,),
                ).fetchall()
        return [dict(row) for row in rows]


def _decode_upload(data: bytes, filename: str | None) -> np.ndarray:
    if not data:
        raise InvalidImageError(f"Empty upload: {filename or 'image'}")
    image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise InvalidImageError(f"Corrupted or unsupported image: {filename or 'image'}")
    return validate_frame(image)


def _person_dict(person: Person) -> dict[str, Any]:
    return person.to_dict()


def _state(request: Request) -> AppState:
    return request.app.state.faceattend


def require_admin(
    x_admin_passphrase: str | None = Header(default=None),
    state: AppState = Depends(_state),
) -> None:
    if state.admin.enabled and not x_admin_passphrase:
        raise HTTPException(status_code=401, detail="X-Admin-Passphrase header is required")
    try:
        if state.admin.enabled:
            state.admin.verify(x_admin_passphrase or "")
    except AuthError as exc:
        raise HTTPException(status_code=401, detail=str(exc)) from exc


def _require_loaded(state: AppState) -> AppState:
    if state.engine is None or state.registration is None or state.recognition is None:
        raise HTTPException(status_code=503, detail="Face model is still loading")
    return state


def _http_error(exc: Exception) -> HTTPException:
    if isinstance(exc, PersonNotFoundError):
        return HTTPException(status_code=404, detail=str(exc))
    if isinstance(exc, (DuplicatePersonError, DuplicateIdentityError)):
        return HTTPException(status_code=409, detail=str(exc))
    if isinstance(exc, (ValidationError, EnrollmentError, InvalidImageError)):
        return HTTPException(status_code=422, detail=str(exc))
    if isinstance(exc, ValueError):
        return HTTPException(status_code=422, detail=str(exc))
    if isinstance(exc, AuthError):
        return HTTPException(status_code=401, detail=str(exc))
    log.exception("Unhandled FaceAttend error")
    return HTTPException(status_code=500, detail="FaceAttend operation failed")


@asynccontextmanager
async def lifespan(app: FastAPI):
    project_root = Path(__file__).resolve().parents[1]
    config_path = os.environ.get(
        "FACEATTEND_CONFIG",
        str(project_root / "config" / "default.yaml"),
    )
    cfg = load_config(config_path)
    if os.environ.get("FACEATTEND_DB_PATH"):
        cfg.storage.db_path = os.environ["FACEATTEND_DB_PATH"]
    elif not Path(cfg.storage.db_path).is_absolute():
        cfg.storage.db_path = str(project_root / cfg.storage.db_path)
    if not Path(cfg.storage.crops_dir).is_absolute():
        cfg.storage.crops_dir = str(project_root / cfg.storage.crops_dir)
    state = AppState(cfg)
    app.state.faceattend = state
    try:
        state.load_model()
        yield
    finally:
        state.close()


app = FastAPI(title="FaceAttend API", version="0.1.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        origin.strip()
        for origin in os.environ.get("FACEATTEND_CORS_ORIGINS", "http://localhost:3000").split(",")
        if origin.strip()
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health(state: AppState = Depends(_state)) -> dict[str, Any]:
    return {
        "status": "ok",
        "model_loaded": state.engine is not None,
        "gallery_size": state.recognition.gallery_size if state.recognition else 0,
    }


@app.post("/auth/login")
def login(credentials: LoginRequest, state: AppState = Depends(_state)) -> dict[str, Any]:
    if credentials.username != "admin":
        raise HTTPException(status_code=401, detail="Invalid admin credentials")
    try:
        state.admin.verify(credentials.password)
    except AuthError as exc:
        raise HTTPException(status_code=401, detail=str(exc)) from exc
    return {
        "access_token": credentials.password,
        "admin": {"username": "admin", "full_name": "Administrator"},
    }


@app.get("/auth/me")
def auth_me(_: None = Depends(require_admin)) -> dict[str, str]:
    return {"username": "admin", "full_name": "Administrator"}


@app.get("/persons")
def list_persons(state: AppState = Depends(_state)) -> list[dict[str, Any]]:
    return [_person_dict(person) for person in state.repo.list_persons()]


@app.post("/persons", status_code=status.HTTP_201_CREATED, dependencies=[Depends(require_admin)])
async def register_person(
    person_id: str = Form(...),
    full_name: str = Form(...),
    department: str | None = Form(default=None),
    class_name: str | None = Form(default=None),
    metadata: str = Form(default="{}"),
    force: bool = Form(default=False),
    images: list[UploadFile] = File(default=[]),
    state: AppState = Depends(_state),
) -> dict[str, Any]:
    state = _require_loaded(state)
    try:
        parsed_metadata = json.loads(metadata)
        if not isinstance(parsed_metadata, dict):
            raise ValueError("metadata must be a JSON object")
        samples: list[EnrollmentSample] = []
        rejected = []
        for upload in images:
            try:
                sample = state.registration.screen_image(
                    _decode_upload(await upload.read(), upload.filename),
                    f"upload:{upload.filename or 'image'}",
                )
            except InvalidImageError as exc:
                rejected.append({"source": upload.filename, "reason": str(exc)})
                continue
            if isinstance(sample, EnrollmentSample):
                samples.append(sample)
            else:
                rejected.append({"source": sample.source, "reason": sample.reason})
        if not samples and not rejected:
            pid = validate_person_id(person_id)
            name = validate_name(full_name)
            if state.repo.person_exists(pid):
                raise DuplicatePersonError(f"Person ID '{pid}' is already registered")
            state.repo.create_person_with_samples(
                pid, name, department, class_name, parsed_metadata, state.engine.model_name, []
            )
            result_person = state.repo.get_person(pid)
            result = None
        else:
            result = state.registration.register(
                person_id,
                full_name,
                samples,
                department,
                class_name,
                parsed_metadata,
                force,
            )
        state.recognition.reload_gallery()
        return {
            "person": _person_dict(result.person if result else result_person),
            "stored_samples": result.stored_samples if result else 0,
            "dropped": ([{"source": item.source, "reason": item.reason} for item in result.dropped]
                        if result else [])
            + rejected,
        }
    except Exception as exc:
        raise _http_error(exc) from exc


@app.post("/persons/{person_id}/face", dependencies=[Depends(require_admin)])
async def add_face_samples(
    person_id: str,
    images: list[UploadFile] = File(...),
    state: AppState = Depends(_state),
) -> dict[str, Any]:
    state = _require_loaded(state)
    try:
        samples: list[EnrollmentSample] = []
        rejected = []
        for upload in images:
            try:
                sample = state.registration.screen_image(
                    _decode_upload(await upload.read(), upload.filename),
                    f"upload:{upload.filename or 'image'}",
                )
            except InvalidImageError as exc:
                rejected.append({"source": upload.filename, "reason": str(exc)})
                continue
            if isinstance(sample, EnrollmentSample):
                samples.append(sample)
            else:
                rejected.append({"source": sample.source, "reason": sample.reason})
        result = state.registration.add_samples(person_id, samples)
        state.recognition.reload_gallery()
        return {
            "person": _person_dict(result.person),
            "stored_samples": result.stored_samples,
            "dropped": [{"source": item.source, "reason": item.reason} for item in result.dropped]
            + rejected,
        }
    except Exception as exc:
        raise _http_error(exc) from exc


@app.patch("/persons/{person_id}", dependencies=[Depends(require_admin)])
def update_person(
    person_id: str,
    full_name: str | None = None,
    department: str | None = None,
    class_name: str | None = None,
    metadata: str | None = None,
    state: AppState = Depends(_state),
) -> dict[str, Any]:
    try:
        fields: dict[str, Any] = {
            key: value
            for key, value in {
                "full_name": full_name,
                "department": department,
                "class_name": class_name,
            }.items()
            if value is not None
        }
        if metadata is not None:
            parsed = json.loads(metadata)
            if not isinstance(parsed, dict):
                raise ValueError("metadata must be a JSON object")
            fields["metadata"] = parsed
        return _person_dict(state.repo.update_person(person_id, **fields))
    except Exception as exc:
        raise _http_error(exc) from exc


@app.delete("/persons/{person_id}", dependencies=[Depends(require_admin)])
def delete_person(person_id: str, state: AppState = Depends(_state)) -> dict[str, str]:
    try:
        crop_paths = state.repo.delete_person(person_id)
        for crop_path in crop_paths:
            Path(crop_path).unlink(missing_ok=True)
        if state.recognition:
            state.recognition.reload_gallery()
        return {"person_id": person_id, "status": "deleted"}
    except Exception as exc:
        raise _http_error(exc) from exc


@app.post("/recognize")
async def recognize(
    image: UploadFile = File(...),
    session_id: str | None = Form(default=None),
    state: AppState = Depends(_state),
) -> list[dict[str, Any]]:
    state = _require_loaded(state)
    try:
        frame = _decode_upload(await image.read(), image.filename)
        if not session_id:
            results = state.recognition.recognize_frame(frame)
            return [result.to_dict() for result in results]

        # Match the desktop live loop: track faces between frames and only
        # expose an identity after enough consistent recognition votes.
        with state.live_trackers_lock:
            now = time.monotonic()
            state.live_trackers = {
                key: value for key, value in state.live_trackers.items()
                if now - value[1] < 300
            }
            tracker_entry = state.live_trackers.get(session_id)
            if tracker_entry is None:
                tracker = FaceTracker(
                    state.cfg.live.track_iou_threshold,
                    state.cfg.live.track_max_missed,
                    state.cfg.live.smoothing_window,
                )
            else:
                tracker = tracker_entry[0]
            state.live_trackers[session_id] = (tracker, now)

        tracks = tracker.update(state.recognition.engine.detect(frame))
        due = [
            track for track in tracks
            # The desktop loop receives every camera frame and recognizes every
            # Nth one. Browser uploads are already sampled, so each upload is
            # the equivalent of a recognition opportunity.
            if track.frames_since_recognition >= 1
        ]
        if due:
            for track, recognition in zip(
                due,
                state.recognition.recognize_faces(frame, [track.face for track in due]),
            ):
                track.add(recognition.result)

        response = []
        for track in tracks:
            decision = track.decide(state.cfg.live.min_votes)
            status_value = (
                "match" if decision.state == TrackState.CONFIRMED
                else "low_quality" if decision.state == TrackState.LOW_QUALITY
                else "unknown"
            )
            response.append({
                "face": track.face.to_dict(),
                "result": {
                    "status": status_value,
                    "person_id": decision.person_id,
                    "full_name": decision.full_name,
                    "similarity": decision.similarity,
                    "detection_score": track.face.score,
                    "quality_reasons": [],
                },
            })
        return response
    except Exception as exc:
        raise _http_error(exc) from exc


@app.websocket("/ws/recognize")
async def recognize_stream(websocket: WebSocket) -> None:
    """Receive JPEG frames and return metadata plus backend-annotated JPEGs."""
    state: AppState = websocket.app.state.faceattend
    session_id = websocket.query_params.get("session_id")
    passphrase = websocket.query_params.get("passphrase")
    if not session_id:
        await websocket.close(code=1008, reason="session_id is required")
        return
    if state.admin.enabled:
        try:
            state.admin.verify(passphrase or "")
        except AuthError:
            await websocket.close(code=1008, reason="authentication required")
            return

    state = _require_loaded(state)
    tracker = FaceTracker(
        state.cfg.live.track_iou_threshold,
        state.cfg.live.track_max_missed,
        state.cfg.live.smoothing_window,
    )
    await websocket.accept()
    try:
        while True:
            payload = await websocket.receive_bytes()
            frame = _decode_upload(payload, "webcam.jpg")
            faces = state.recognition.engine.detect(frame)
            tracks = tracker.update(faces)
            due = [track for track in tracks if track.frames_since_recognition >= 1]
            if due:
                recognitions = state.recognition.recognize_faces(
                    frame, [track.face for track in due]
                )
                for track, recognition in zip(due, recognitions):
                    track.add(recognition.result)

            response_faces = []
            for track in tracks:
                decision = track.decide(state.cfg.live.min_votes)
                status_value = (
                    "match" if decision.state == TrackState.CONFIRMED
                    else "low_quality" if decision.state == TrackState.LOW_QUALITY
                    else "unknown"
                )
                draw_track(frame, track.face.bbox, track.face.score, decision)
                response_faces.append({
                    "bbox": track.face.to_dict()["bbox"],
                    "recognized": decision.state == TrackState.CONFIRMED,
                    "person_id": decision.person_id,
                    "full_name": decision.full_name
                    or ("Low quality" if status_value == "low_quality" else "Unknown"),
                    "confidence": decision.similarity or 0,
                    "status": status_value,
                })
            draw_status(
                frame,
                f"faces {len(faces)} | gallery {state.recognition.gallery_size}",
            )
            encoded, output = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if not encoded:
                raise InvalidImageError("Could not encode processed webcam frame")
            await websocket.send_json({"faces": response_faces})
            await websocket.send_bytes(output.tobytes())
    except WebSocketDisconnect:
        log.info("Recognition stream disconnected: %s", session_id)
    except Exception:
        log.exception("Recognition stream failed: %s", session_id)
        try:
            await websocket.close(code=1011, reason="recognition failed")
        except RuntimeError:
            pass


@app.get("/attendance")
def list_attendance(
    person_id: str | None = None,
    limit: int = 100,
    state: AppState = Depends(_state),
) -> list[dict[str, Any]]:
    if not 1 <= limit <= 500:
        raise HTTPException(status_code=422, detail="limit must be between 1 and 500")
    return state.attendance.list(person_id, limit)


@app.post("/attendance", status_code=status.HTTP_201_CREATED, dependencies=[Depends(require_admin)])
def create_attendance(
    event: AttendanceCreate,
    state: AppState = Depends(_state),
) -> dict[str, Any]:
    try:
        state.repo.get_person(event.person_id)
        return state.attendance.create(event)
    except Exception as exc:
        raise _http_error(exc) from exc
