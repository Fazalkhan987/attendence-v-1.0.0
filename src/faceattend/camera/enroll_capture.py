"""Interactive webcam capture for enrollment (auto-captures good frames)."""
from __future__ import annotations

import time

import cv2

from ..config import LiveConfig
from ..services.registration import EnrollmentSample, Rejection, RegistrationService
from .overlay import draw_label, draw_status
from .webcam import Webcam

WINDOW = "FaceAttend - enrollment (SPACE pause/resume, q finish/abort)"
PROMPTS = ["Look straight at the camera", "Turn slightly left", "Turn slightly right",
           "Tilt chin slightly up", "Tilt chin slightly down", "Neutral expression",
           "Smile", "Look straight again"]


def capture_samples(service: RegistrationService, live_cfg: LiveConfig, target: int) -> list[EnrollmentSample]:
    samples: list[EnrollmentSample] = []
    interval = service.cfg.auto_capture_interval_s
    last_capture, paused, status = 0.0, False, ""
    with Webcam(live_cfg) as cam:
        cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
        try:
            while len(samples) < target:
                frame = cam.read()
                if frame is None:
                    continue
                view = frame.copy()
                now = time.monotonic()
                if not paused and now - last_capture >= interval:
                    out = service.screen_image(frame, f"webcam:{len(samples) + 1}")
                    if isinstance(out, EnrollmentSample):
                        dup = any(float(s.embedding @ out.embedding) >= service.cfg.near_duplicate_threshold
                                  for s in samples)
                        if dup:
                            status = "Move a little - frame too similar to previous"
                        else:
                            samples.append(out)
                            status = f"Captured {len(samples)}/{target}"
                            last_capture = now
                    elif isinstance(out, Rejection):
                        status = f"Rejected: {out.reason}"
                prompt = PROMPTS[len(samples) % len(PROMPTS)]
                draw_label(view, 10, 60, [prompt, status or "Hold still"], (90, 90, 90))
                draw_status(view, f"{len(samples)}/{target} samples" + (" | PAUSED" if paused else ""))
                cv2.imshow(WINDOW, view)
                key = cv2.waitKey(1) & 0xFF
                if key == ord(" "):
                    paused = not paused
                elif key in (ord("q"), 27):
                    break
        finally:
            cv2.destroyAllWindows()
    return samples
