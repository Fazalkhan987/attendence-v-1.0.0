"""Person record management that needs no ML model (update, delete, read)."""
from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Any

from ..storage.database import Person, Repository
from ..validation import clean_optional, validate_name, validate_person_id
from .registration import remove_files

log = logging.getLogger(__name__)


class PersonService:
    def __init__(self, repo: Repository, crops_dir: str | Path):
        self.repo = repo
        self.crops_dir = Path(crops_dir)

    def get(self, person_id: str) -> Person:
        return self.repo.get_person(validate_person_id(person_id))

    def list(self) -> list[Person]:
        return self.repo.list_persons()

    def update(self, person_id: str, full_name: str | None = None, department: str | None = None,
               class_name: str | None = None) -> Person:
        pid = validate_person_id(person_id)
        fields: dict[str, Any] = {}
        if full_name is not None:
            fields["full_name"] = validate_name(full_name)
        if department is not None:
            fields["department"] = clean_optional(department, "department")
        if class_name is not None:
            fields["class_name"] = clean_optional(class_name, "class")
        return self.repo.update_person(pid, **fields)

    def delete(self, person_id: str) -> None:
        """Remove the record, every embedding, any stored crops, and the crop folder."""
        pid = validate_person_id(person_id)
        remove_files(self.repo.delete_person(pid))
        person_dir = self.crops_dir / pid
        if person_dir.is_dir():
            shutil.rmtree(person_dir, ignore_errors=True)
        log.info("Deleted %s and all associated biometric data", pid)
