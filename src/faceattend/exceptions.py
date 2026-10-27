"""Exception hierarchy. Callers (CLI today, FastAPI later) map these to user-facing errors."""


class FaceAttendError(Exception):
    """Base class for all application errors."""


class ConfigError(FaceAttendError):
    pass


class ModelLoadError(FaceAttendError):
    pass


class InvalidImageError(FaceAttendError):
    pass


class CameraError(FaceAttendError):
    pass


class ValidationError(FaceAttendError):
    pass


class PersonNotFoundError(FaceAttendError):
    pass


class DuplicatePersonError(FaceAttendError):
    """The person ID is already registered."""


class DuplicateIdentityError(FaceAttendError):
    """The face appears to belong to an already-registered person."""

    def __init__(self, existing_person_id: str, similarity: float):
        self.existing_person_id = existing_person_id
        self.similarity = similarity
        super().__init__(
            f"Face closely matches already-registered person '{existing_person_id}' "
            f"(similarity {similarity:.3f}). Use --force only if you have verified "
            f"this is a different person."
        )


class EnrollmentError(FaceAttendError):
    pass


class AuthError(FaceAttendError):
    pass


class EvaluationDataError(FaceAttendError):
    """Evaluation dataset is missing, malformed or leaks between splits."""
