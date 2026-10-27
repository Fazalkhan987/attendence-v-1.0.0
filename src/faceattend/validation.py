"""Input validation for person records."""
from __future__ import annotations

import re

from .exceptions import ValidationError

_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.\-]{0,63}$")


def validate_person_id(person_id: str) -> str:
    pid = (person_id or "").strip()
    if not _ID_RE.match(pid):
        raise ValidationError(
            "Invalid person ID: use 1-64 chars, letters/digits/'_'/'-'/'.', starting with a letter or digit"
        )
    return pid


def validate_name(name: str) -> str:
    n = " ".join((name or "").split())
    if not n or len(n) > 120:
        raise ValidationError("Full name must be 1-120 characters")
    return n


def clean_optional(value: str | None, field: str, max_len: int = 120) -> str | None:
    if value is None:
        return None
    v = " ".join(value.split())
    if len(v) > max_len:
        raise ValidationError(f"{field} must be at most {max_len} characters")
    return v or None
