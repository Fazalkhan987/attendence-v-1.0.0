# FaceAttend — Face Detection & Recognition Engine

Core AI engine for a future Smart Attendance System. Standalone Python application (CLI + OpenCV window); every capability lives in UI-independent services so a FastAPI backend can call them later. No FastAPI/React code is included yet, by design.

## 1. Pipeline choice

| Stage | Choice | Why |
|---|---|---|
| Detection | **SCRFD-10GF** (InsightFace `buffalo_l`, ONNX) | Strong accuracy/speed trade-off, multi-face, returns 5 landmarks needed for ArcFace alignment |
| Alignment | `insightface.utils.face_align.norm_crop` | The exact 5-point similarity transform to the 112×112 template the model was trained with |
| Recognition | **ArcFace ResNet-50 trained on WebFace600K** (`w600k_r50.onnx`), 512-d | Well-established margin-based embedding; new people are enrolled without retraining |
| Inference | ONNX Runtime (CPU or CUDA) | No PyTorch needed; same models on Windows/Linux/macOS |
| Matching | Cosine similarity, open-set rule: threshold **and** margin over runner-up | Unknown people are rejected instead of being forced onto the nearest identity |

One recognition framework only (no FaceNet mixed in): detector and recognizer come from one maintained, version-pinned model pack, which avoids preprocessing mismatches.

> ### ⚠ Licence — read before any commercial deployment
> InsightFace **code** is MIT. The **pretrained models** (including `buffalo_l`) are licensed by InsightFace for **non-commercial research only**. This project is fine for university/research use as-is. A commercial attendance product needs a commercial licence from InsightFace or a different model with a permissive licence. The architecture makes swapping easy: implement `FaceDetector` / `FaceEmbedder` (see §8).

## 2. Requirements & compatibility

- Python **3.10 – 3.12** (tested on **3.12.3**, Linux x86_64).
- Pinned/tested: `insightface==0.7.3`, `numpy 1.26.4` (kept `<2.0`), `opencv-python 4.10.0.84`, `onnxruntime 1.31.0`.
- `insightface 0.7.3` ships as **source only** and compiles a small C++/Cython extension, so a compiler is required:
  - **Windows:** "Microsoft C++ Build Tools" → workload *Desktop development with C++*.
  - **Ubuntu/Debian:** `sudo apt install -y build-essential python3-dev python3-venv`
  - **macOS:** `xcode-select --install`
- InsightFace pulls in `albumentations`, which installs **`opencv-python-headless`** alongside `opencv-python`. Both provide `cv2`; the headless one breaks `cv2.imshow`. Step 3 below fixes it.
- A newer `insightface 1.0` exists on PyPI; this project was built and tested against 0.7.3 only.

## 3. Installation

### Linux / macOS
```bash
cd face_attendance
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements-dev.txt
pip uninstall -y opencv-python-headless
pip install --force-reinstall --no-deps "opencv-python==4.10.0.84"
pip install -e . --no-deps  
```

### Windows (PowerShell)
```powershell
cd face_attendance
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements-dev.txt
pip uninstall -y opencv-python-headless
pip install --force-reinstall --no-deps "opencv-python==4.10.0.84"
pip install -e . --no-deps
```

### GPU (optional, NVIDIA)
```bash
pip uninstall -y onnxruntime
pip install onnxruntime-gpu
```
Then in `config/default.yaml`: `providers: [CUDAExecutionProvider, CPUExecutionProvider]` and `ctx_id: 0`. The CUDA/cuDNN versions must match your `onnxruntime-gpu` build (see the ONNX Runtime CUDA compatibility table).

### Models
Downloaded automatically (~280 MB zip, from the InsightFace GitHub release) into `~/.insightface/models/buffalo_l/` the first time the engine loads:
```bash
faceattend doctor
```
Offline/manual: download `buffalo_l.zip` from the InsightFace v0.7 release, unzip to `~/.insightface/models/buffalo_l/` (it must contain `det_10g.onnx` and `w600k_r50.onnx`), or point `model.root` in the config at another directory containing `models/buffalo_l/`.

## 4. Quick start

```bash
faceattend init-admin                                   # set admin passphrase (once)
faceattend doctor --camera                              # versions, providers, model, camera
faceattend register --id S001 --name "Ayesha Malik" --department CS --class BSCS-7A
faceattend register --id S002 --name "Bilal Ahmed" --source dir --dir ./photos/bilal
faceattend list
faceattend live                                         # q / Esc to quit
faceattend recognize ./some_photo.jpg                   # JSON output
faceattend add-samples --id S001                        # more samples later
faceattend update --id S001 --name "Ayesha R. Malik"
faceattend delete --id S002                             # removes record + all embeddings (+ crops)
```
`python -m faceattend ...` works too. For scripts, the admin passphrase can come from `FACEATTEND_ADMIN_PASSWORD` instead of the prompt.

**Webcam enrollment:** the window auto-captures a frame every 0.7 s when exactly one face passes the strict quality checks, prompting the person to turn slightly left/right, tilt, smile, etc. Near-identical frames (cosine ≥ 0.97 to an existing sample) are skipped, so the person must actually move. SPACE pauses, `q` finishes.

## 5. How registration works

1. Each image: decode (corrupt files rejected) → detect → **exactly one face** required (0 or ≥2 → rejected) → align → quality check → embed.
2. **Quality checks** (`enrollment_quality`): face size, detector confidence, blur (Laplacian variance), brightness, contrast, and a 5-landmark pose heuristic (roll, yaw, pitch). The pose check is a geometric approximation, not a calibrated head-pose estimator; smiles raise the pitch ratio.
3. **Sample set checks:** near-duplicates dropped; samples inconsistent with the set's medoid (cosine < 0.35) dropped as a possible different person.
4. At least `min_samples` (5) must survive, otherwise **nothing is saved** (single transaction).
5. **Duplicate identity:** the new person's mean embedding is compared with every enrolled person; ≥ 0.55 blocks registration unless `--force` (twins / look-alikes after manual verification).
6. Stored per sample: 512 float32 values as a raw BLOB, the model name, source tag and quality metrics. **No images** unless `store_aligned_crops: true` (only 112×112 aligned crops then — useful for re-embedding after a model change, at a privacy cost).

**Combining samples:** default `match_strategy: centroid` compares a probe with the L2-normalised mean of the person's samples. `max` (best single sample) and `topk_mean` are available; compare them with `faceattend evaluate`.

## 6. Recognition decision rule

For each face: detect → recognition-quality check (looser than enrollment) → embed → score every identity → accept the top-1 only if `similarity ≥ threshold` **and** `top1 − top2 ≥ margin`. Otherwise **Unknown**. Faces failing quality are **Low quality** (no identity claimed).

Live mode tracks faces with IoU, re-runs recognition every N frames per track, and only shows a name after `min_votes` (4) of the last `smoothing_window` (7) decisions agree. Until then the box says *Verifying…*. An identity-confirmed callback fires once per track — the hook for a future attendance module.

## 7. Thresholds and evaluation

`similarity_threshold: 0.45` is a **starting point, not a calibrated value**. Calibrate on your own cameras and people:

```
eval_data/
  val/                     # choose the threshold here
    gallery/<person>/*.jpg       enrollment images
    probe/<person>/*.jpg         DIFFERENT captures of the same people (other day/lighting)
    impostor/<person>/*.jpg      people NOT in gallery
    conditions.csv               optional: relative_path,condition   e.g. probe/s01/a.jpg,low_light
    detection/ + detection_annotations.json   optional: {"img.jpg": [[x1,y1,x2,y2], ...]}
  test/                    # report final numbers here, with the threshold fixed
```
```bash
faceattend evaluate --data eval_data/val --target-far 0.001
faceattend evaluate --data eval_data/test --threshold 0.52 --other-split eval_data/val
```
Reported: genuine/impostor score distributions, EER, threshold at target FAR, TAR/FRR/FAR sweep, open-set identification accuracy, misidentification rate, unknown-rejection rate (FPIR), failure-to-acquire, per-condition breakdown, detection precision/recall (with annotations), latency mean/p50/p95, hardware and ONNX providers. Outputs `reports/eval-*.json` and `.md`.

Leakage protection: evaluation builds its own in-memory gallery (never touches the production DB); byte-identical images between gallery and probe/impostor abort the run; impostor identities that also appear in the gallery abort the run; `--other-split` flags images shared between val and test. Same-session near-duplicates (e.g. consecutive video frames) cannot be detected automatically — capture probes on a different day.

The report warns when the requested FAR is below what the data can measure (FAR resolution = 1 / impostor comparisons). For attendance, pick the operating point by FAR (wrongly marking someone present) first, then accept the resulting FRR (people who must retry).

## 8. Architecture

```
src/faceattend/
  config.py            typed YAML config, unknown keys rejected
  schemas.py           DetectedFace, QualityReport, MatchResult (JSON-friendly)
  detection/           base.py (FaceDetector protocol) · scrfd.py
  recognition/         base.py (FaceEmbedder protocol) · arcface.py · matcher.py (gallery + open-set rule)
  quality.py           blur / exposure / size / pose checks
  storage/database.py  SQLite repository (WAL, FK cascade, secure_delete, audit log)
  security.py          PBKDF2 admin passphrase
  services/
    engine.py          FaceEngine: detect → align → quality → embed (no UI)
    registration.py    screening, consistency & duplicate checks, register / add samples
    persons.py         read / update / delete (no model needed)
    recognition.py     recognize_frame(), reload_gallery()
  camera/              webcam.py · tracker.py · live.py · enroll_capture.py · overlay.py
  liveness/base.py     anti-spoofing interface (PLANNED, not implemented)
  evaluation/          dataset.py · metrics.py · runner.py · report.py
  cli.py
```

**Swapping models:** write a class with `detect(image) -> list[DetectedFace]` or `align()` + `embed()` and build `FaceEngine(detector, embedder)`. Embeddings are tagged with the model name, so a new recognizer ignores old embeddings (and logs a warning) instead of mixing incompatible vectors — re-enrollment is required.

### FastAPI and React integration

The working web application is in this project's `backend/` and `frontend/`
directories.
Its FastAPI backend imports this package's `build_engine`, `Repository`,
`RegistrationService`, and `RecognitionService`, while the React frontend
captures enrollment samples and consumes the REST APIs. The live camera uses
`/ws/recognize`: the browser sends JPEG frames as binary WebSocket messages,
and the backend returns recognition metadata followed by a backend-annotated
JPEG frame. The displayed live image is therefore the processed backend frame,
with the same tracking and vote smoothing as `faceattend live`.

From `backend`, install the local package and run:

```bash
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

The backend loads this package's `config/default.yaml` and `data/faceattend.db`
by default. Set `FACEATTEND_CONFIG` or `FACEATTEND_DB_PATH` to override them.

```python
engine = build_engine(cfg)                                    # once at startup
repo = Repository(cfg.storage.db_path)
recog = RecognitionService(engine, repo, cfg.recognition, QualityAssessor(cfg.recognition_quality))
reg = RegistrationService(engine, repo, cfg.enrollment, QualityAssessor(cfg.enrollment_quality), cfg.storage.crops_dir)

# POST /recognize  -> [r.to_dict() for r in recog.recognize_frame(decode(upload))]
# POST /persons    -> screen each upload with reg.screen_image(), then reg.register(...); recog.reload_gallery()
# DELETE /persons/{id} -> PersonService(repo, crops_dir).delete(id); recog.reload_gallery()
```
Exceptions map cleanly to HTTP codes (`PersonNotFoundError`→404, `DuplicatePersonError`/`DuplicateIdentityError`→409, `ValidationError`/`EnrollmentError`/`InvalidImageError`→422, `AuthError`→401/403). ONNX inference is CPU-bound: run it in a threadpool or a single worker process.

## 9. Security & privacy

Implemented: admin passphrase (PBKDF2-SHA256, 600k iterations) for register/add/update/delete; DB file `0600` and data directory `0700` on POSIX; SQLite `secure_delete` plus WAL checkpoint on deletion; no pickle anywhere; logs and audit entries never contain embeddings or images; images not stored by default; full deletion of a person's data.

Not implemented — your responsibility in deployment: encryption at rest (use full-disk encryption, or SQLCipher), OS-level access control on the machine, backups policy, consent records and retention rules under applicable law, TLS/authentication once an API exists. Face embeddings are biometric data; treat the DB like a password store.

## 10. Feature status

| Feature | Status |
|---|---|
| SCRFD multi-face detection, landmarks, configurable threshold | Implemented, tested with real model |
| ArcFace alignment + 512-d embeddings | Implemented, tested with real model |
| Registration (webcam / directory), quality gates, duplicate checks, update, add samples, delete | Implemented; directory flow tested end-to-end with real model; webcam capture loop not tested on a physical camera |
| Open-set matching (threshold + margin), multi-face frames | Implemented, unit + real-model tested |
| Live webcam with tracking & vote smoothing | Implemented; loop tested headless with mocked camera/display, **not on a physical camera** |
| Evaluation & reporting | Implemented, tested end-to-end on a tiny synthetic set |
| Anti-spoofing / liveness | **Planned only** — interface + placeholder returning `not_checked` |
| Encryption at rest | Not implemented (see §9) |
| FastAPI / React | Implemented in sibling `attendance-system/` |

## 11. Measured results (so far) and limitations

What was actually measured in the build environment (Linux, 1 vCPU, CPU provider, Python 3.12.3), on InsightFace's bundled sample group photo only:
- 6 faces detected in a 1280×886 image; SCRFD at 640×640 ≈ 270–290 ms per image; align+quality+embed ≈ 320 ms per face (single core — expect much faster on a normal laptop CPU and far faster on GPU).
- Cross-person cosine similarity among those 6 people: max 0.21. Mirrored/darkened/blurred copies of  = Repository(cfg.storage.db_path)
recog = RecognitionService(engine, repo, cfg.recognition, QualityAssessor(cfg.recognition_quality))
reg = Registrthe same faces: ≥ 0.93.

These are **smoke-test numbers, not accuracy benchmarks**. No accuracy, FAR or FRR claim is made for your deployment until you run `faceattend evaluate` on representative data.

Known limitations: accuracy degrades with strong profile views, heavy occlusion (masks, hands), very low light, motion blur and small/distant faces; the pose check is heuristic; a plain webcam cannot stop photo/screen replay attacks; known demographic performance differences exist in face recognition generally — measure on your own population; SQLite with a single process is assumed.

## 12. Tests

```bash
pytest                                                  # 32 unit tests, no models needed (fakes)
pytest -m integration                                   # real SCRFD/ArcFace checks (needs buffalo_l)
FACEATTEND_MODEL_ROOT=/path/to/root pytest -m integration
```
