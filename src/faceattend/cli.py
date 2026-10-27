"""Command-line entry point: `faceattend <command>` (or `python -m faceattend`)."""
from __future__ import annotations

import argparse
import getpass
import json
import logging
import sys

from .config import AppConfig, load_config
from .exceptions import DuplicateIdentityError, FaceAttendError
from .logging_setup import setup_logging
from .quality import QualityAssessor
from .security import AdminGuard
from .storage.database import Repository

log = logging.getLogger("faceattend")


class Context:
    """Lazily builds the heavy objects so cheap commands (list/update) don't load models."""

    def __init__(self, cfg: AppConfig):
        self.cfg = cfg
        self.repo = Repository(cfg.storage.db_path)
        self.guard = AdminGuard(self.repo, cfg.security.require_admin_auth)
        self._engine = None

    @property
    def engine(self):
        if self._engine is None:
            from .services.engine import build_engine
            self._engine = build_engine(self.cfg)
        return self._engine

    def registration(self):
        from .services.registration import RegistrationService
        return RegistrationService(self.engine, self.repo, self.cfg.enrollment,
                                   QualityAssessor(self.cfg.enrollment_quality), self.cfg.storage.crops_dir)

    def persons(self):
        from .services.persons import PersonService
        return PersonService(self.repo, self.cfg.storage.crops_dir)

    def recognition(self):
        from .services.recognition import RecognitionService
        return RecognitionService(self.engine, self.repo, self.cfg.recognition,
                                  QualityAssessor(self.cfg.recognition_quality))

    def admin(self) -> None:
        self.guard.require(lambda: getpass.getpass("Admin passphrase: "))


def _collect_samples(ctx: Context, svc, args) -> list:
    if args.source == "dir":
        if not args.dir:
            raise FaceAttendError("--dir is required when --source dir")
        res = svc.screen_directory(args.dir)
        for r in res.rejected:
            print(f"  rejected {r.source}: {r.reason}")
        print(f"  accepted {len(res.accepted)} image(s)")
        return res.accepted
    from .camera.enroll_capture import capture_samples
    return capture_samples(svc, ctx.cfg.live, args.samples or ctx.cfg.enrollment.target_samples)


def cmd_init_admin(ctx: Context, args) -> None:
    current = None
    if ctx.guard.is_initialised:
        current = getpass.getpass("Current admin passphrase: ")
    pw = getpass.getpass("New admin passphrase (min 10 chars): ")
    if pw != getpass.getpass("Repeat: "):
        raise FaceAttendError("Passphrases do not match")
    ctx.guard.set_password(pw, current)
    print("Admin passphrase set.")


def cmd_register(ctx: Context, args) -> None:
    ctx.admin()
    svc = ctx.registration()
    if ctx.repo.person_exists(args.id.strip()):
        raise FaceAttendError(f"Person ID '{args.id}' already exists (use add-samples or update)")
    samples = _collect_samples(ctx, svc, args)
    try:
        res = svc.register(args.id, args.name, samples, args.department, args.class_name, force=args.force)
    except DuplicateIdentityError as exc:
        raise FaceAttendError(str(exc)) from exc
    for d in res.dropped:
        print(f"  dropped {d.source}: {d.reason}")
    print(f"Registered {res.person.person_id} ({res.person.full_name}) with {res.stored_samples} samples.")


def cmd_add_samples(ctx: Context, args) -> None:
    ctx.admin()
    svc = ctx.registration()
    ctx.repo.get_person(args.id.strip())   # fail fast before opening the camera
    res = svc.add_samples(args.id, _collect_samples(ctx, svc, args))
    for d in res.dropped:
        print(f"  dropped {d.source}: {d.reason}")
    print(f"Added {res.stored_samples} samples; {res.person.person_id} now has {res.person.sample_count}.")


def cmd_update(ctx: Context, args) -> None:
    ctx.admin()
    p = ctx.persons().update(args.id, args.name, args.department, args.class_name)
    print(json.dumps(p.to_dict(), indent=2))


def cmd_delete(ctx: Context, args) -> None:
    ctx.admin()
    if not args.yes and input(f"Permanently delete '{args.id}' and all biometric data? [y/N] ").lower() != "y":
        print("Aborted.")
        return
    ctx.persons().delete(args.id)
    print(f"Deleted {args.id}.")


def cmd_list(ctx: Context, args) -> None:
    persons = ctx.repo.list_persons()
    if args.json:
        print(json.dumps([p.to_dict() for p in persons], indent=2))
        return
    if not persons:
        print("No persons registered.")
    for p in persons:
        extra = " | ".join(x for x in (p.department, p.class_name) if x)
        print(f"{p.person_id:<16} {p.full_name:<30} samples={p.sample_count:<3} {extra}")


def cmd_live(ctx: Context, args) -> None:
    from .camera.live import run_live
    if args.camera is not None:
        ctx.cfg.live.camera_index = args.camera
    run_live(ctx.recognition(), ctx.cfg.live)


def cmd_recognize(ctx: Context, args) -> None:
    from .image_utils import read_image
    svc = ctx.recognition()
    results = svc.recognize_frame(read_image(args.image))
    print(json.dumps([r.to_dict() for r in results], indent=2))


def cmd_evaluate(ctx: Context, args) -> None:
    from .evaluation.dataset import check_cross_split_leakage, load_split
    from .evaluation.report import write_report
    from .evaluation.runner import evaluate_split
    split = load_split(args.data)
    if args.other_split:
        split.warnings += check_cross_split_leakage(load_split(args.other_split), split)
    report = evaluate_split(ctx.engine, ctx.cfg, split, args.threshold, args.target_far)
    jp, mp = write_report(report, ctx.cfg.storage.reports_dir)
    print(mp.read_text(encoding="utf-8"))
    print(f"Saved {jp} and {mp}")


def cmd_doctor(ctx: Context, args) -> None:
    import platform
    print(f"Python      {platform.python_version()} ({platform.system()})")
    import cv2
    import numpy
    gui = next((ln.split(":", 1)[1].strip() for ln in cv2.getBuildInformation().splitlines()
                if ln.strip().startswith("GUI:")), "unknown")
    print(f"numpy       {numpy.__version__}\nopencv      {cv2.__version__} (GUI backend: {gui or 'NONE - headless build'})")
    try:
        import onnxruntime as ort
        print(f"onnxruntime {ort.__version__} providers={ort.get_available_providers()}")
    except ImportError:
        print("onnxruntime NOT INSTALLED")
    eng = ctx.engine
    print(f"model       {eng.model_name}")
    print(f"database    {ctx.repo.path} ({len(ctx.repo.list_persons())} persons)")
    print(f"admin auth  {'set' if ctx.guard.is_initialised else 'NOT SET - run init-admin'}")
    if args.camera:
        from .camera.webcam import Webcam
        with Webcam(ctx.cfg.live) as cam:
            f = cam.read()
            print(f"camera      OK {None if f is None else f.shape}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="faceattend", description="Face registration and recognition engine")
    p.add_argument("--config", help="YAML config path (default: config/default.yaml)")
    sub = p.add_subparsers(dest="command", required=True)

    sub.add_parser("init-admin", help="set or change the admin passphrase")

    def enroll_args(sp):
        sp.add_argument("--source", choices=["webcam", "dir"], default="webcam")
        sp.add_argument("--dir", help="directory of images (with --source dir)")
        sp.add_argument("--samples", type=int, help="webcam samples to capture")

    r = sub.add_parser("register", help="register a new person")
    r.add_argument("--id", required=True)
    r.add_argument("--name", required=True)
    r.add_argument("--department")
    r.add_argument("--class", dest="class_name")
    r.add_argument("--force", action="store_true", help="register even if the face matches an existing person")
    enroll_args(r)

    a = sub.add_parser("add-samples", help="add face samples to an existing person")
    a.add_argument("--id", required=True)
    enroll_args(a)

    u = sub.add_parser("update", help="update a person's details")
    u.add_argument("--id", required=True)
    u.add_argument("--name")
    u.add_argument("--department")
    u.add_argument("--class", dest="class_name")

    d = sub.add_parser("delete", help="delete a person and all their biometric data")
    d.add_argument("--id", required=True)
    d.add_argument("--yes", action="store_true")

    ls = sub.add_parser("list", help="list registered persons")
    ls.add_argument("--json", action="store_true")

    lv = sub.add_parser("live", help="run real-time webcam recognition")
    lv.add_argument("--camera", type=int)

    rc = sub.add_parser("recognize", help="recognise faces in one image (prints JSON)")
    rc.add_argument("image")

    ev = sub.add_parser("evaluate", help="evaluate on a labelled split (see README)")
    ev.add_argument("--data", required=True, help="split directory, e.g. eval_data/val")
    ev.add_argument("--threshold", type=float, help="threshold to report at (default: chosen at --target-far)")
    ev.add_argument("--target-far", type=float, default=1e-3)
    ev.add_argument("--other-split", help="another split to check for image leakage against")

    dr = sub.add_parser("doctor", help="check environment, models, database")
    dr.add_argument("--camera", action="store_true", help="also test the camera")
    return p


COMMANDS = {
    "init-admin": cmd_init_admin, "register": cmd_register, "add-samples": cmd_add_samples,
    "update": cmd_update, "delete": cmd_delete, "list": cmd_list, "live": cmd_live,
    "recognize": cmd_recognize, "evaluate": cmd_evaluate, "doctor": cmd_doctor,
}


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        cfg = load_config(args.config)
        setup_logging(cfg.logging)
        ctx = Context(cfg)
        try:
            COMMANDS[args.command](ctx, args)
        finally:
            ctx.repo.close()
        return 0
    except FaceAttendError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    sys.exit(main())
