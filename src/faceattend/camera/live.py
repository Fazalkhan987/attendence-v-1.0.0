"""Real-time webcam recognition loop."""
from __future__ import annotations

import logging
import time
from typing import Callable

import cv2

from ..config import LiveConfig
from ..exceptions import CameraError
from ..liveness.base import LivenessChecker, NullLivenessChecker
from ..services.recognition import RecognitionService
from .overlay import draw_status, draw_track
from .tracker import FaceTracker, TrackState
from .webcam import Webcam

log = logging.getLogger(__name__)
WINDOW = "FaceAttend - live recognition (q/Esc to quit)"

# Called once per track when an identity becomes CONFIRMED. A future attendance
# module would plug in here (with de-duplication per session and a liveness gate).
ConfirmedCallback = Callable[[str, str | None, float, float], None]


def run_live(service: RecognitionService, cfg: LiveConfig,
             on_confirmed: ConfirmedCallback | None = None,
             liveness: LivenessChecker | None = None) -> None:
    liveness = liveness or NullLivenessChecker()
    tracker = FaceTracker(cfg.track_iou_threshold, cfg.track_max_missed, cfg.smoothing_window)
    if service.gallery_size == 0:
        log.warning("Gallery is empty - every face will be shown as Unknown. Register people first.")
    fps, last = 0.0, time.perf_counter()
    with Webcam(cfg) as cam:
        cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
        try:
            while True:
                frame = cam.read()
                if frame is None:
                    continue
                faces = service.engine.detect(frame)
                tracks = tracker.update(faces)
                due = [t for t in tracks if t.frames_since_recognition >= cfg.recognize_every_n_frames]
                if due:
                    for t, rec in zip(due, service.recognize_faces(frame, [t.face for t in due])):
                        t.add(rec.result)
                for t in tracks:
                    d = t.decide(cfg.min_votes)
                    if d.state == TrackState.CONFIRMED and t.confirmed_person != d.person_id:
                        t.confirmed_person = d.person_id
                        live = liveness.check(frame, t.face).status.value
                        log.info("Confirmed %s (sim %.3f, liveness=%s)", d.person_id, d.similarity, live)
                        if on_confirmed:
                            on_confirmed(d.person_id, d.full_name, d.similarity, time.time())
                    draw_track(frame, t.face.bbox, t.face.score, d)
                now = time.perf_counter()
                fps = 0.9 * fps + 0.1 * (1.0 / max(now - last, 1e-6))
                last = now
                draw_status(frame, f"FPS {fps:.1f} | faces {len(faces)} | gallery {service.gallery_size}"
                                   " | liveness: not checked")
                cv2.imshow(WINDOW, frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                    break
        except CameraError:
            log.error("Camera stream lost")
            raise
        finally:
            cv2.destroyAllWindows()
