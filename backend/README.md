# FaceAttend FastAPI backend

This backend is an isolated adapter around the existing `faceattend` package.
It does not copy or modify the model implementation. The package's
`Repository` stores people and embeddings in SQLite; this API adds an
`attendance_events` table to that same database.

## Install and run

From this `backend/` directory, using the Python 3.10-3.12 environment already
used by the project:

```bash
python -m pip install -r requirements.txt
python -m pip install -e ..
uvicorn main:app --reload --port 8000
```

The first startup loads InsightFace and may download the configured
`buffalo_l` model pack. By default it reads `../config/default.yaml` and uses
`../data/faceattend.db`, regardless of whether Uvicorn is started from the
project root or this directory. Set `FACEATTEND_CONFIG` to use a different
YAML config, or `FACEATTEND_DB_PATH` to override the database path. For a
separate frontend origin, set `FACEATTEND_CORS_ORIGINS` to a comma-separated
allowlist.

The API uses the package's admin passphrase for biometric writes. Initialise it
once from the project root:

```bash
faceattend init-admin
```

Send that passphrase in `X-Admin-Passphrase` for `POST /persons` and
`POST /attendance`. The API never stores uploaded images; only accepted
embeddings and quality metadata are persisted, following the package defaults.

## Endpoints

- `GET /health` - model and gallery status.
- `GET /persons` - registered people and sample counts.
- `POST /persons` - multipart form with `person_id`, `full_name`, optional
  `department`, `class_name`, `metadata` (JSON object), `force`, and one or
  more `images`. The frontend may first create the person without images and
  then use the face endpoint below. Direct registration with images requires
  the configured minimum sample count (five by default).
- `POST /persons/{person_id}/face` - multipart `images` endpoint for adding
  enrollment samples to an existing person.
- `PATCH /persons/{person_id}` - update person fields. Send optional
  `full_name`, `department`, `class_name`, and `metadata` query/form fields.
- `DELETE /persons/{person_id}` - permanently delete the person, embeddings,
  and any configured aligned crops.
- `POST /recognize` - multipart upload named `image`; returns one result per
  detected face, including `match`, `unknown`, or `low_quality`.
- `POST /attendance` - JSON `{ "person_id": "...", "event_type": "check-in" }`.
- `GET /attendance?person_id=S001&limit=100` - newest attendance events.

Interactive API documentation is available at `http://localhost:8000/docs`.

## React integration

Use `fetch` or Axios with `multipart/form-data` for `/persons` and
`/recognize`; do not set the multipart `Content-Type` header manually because
the browser supplies the boundary. After a successful recognition, only create
an attendance event when the returned result has `status === "match"` and use
its `person_id`.

For production, put the service behind TLS and application authentication,
restrict CORS to the deployed frontend origin, and define retention/consent
rules for biometric data. The package's SQLite repository is intended for a
single API worker; use one worker or migrate persistence before running
multiple processes.
