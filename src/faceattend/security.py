"""Admin authentication for operations that create, modify or delete biometric data.

A PBKDF2-HMAC-SHA256 hash of the admin passphrase is stored in the settings table.
This protects against casual misuse of the CLI; it is NOT a substitute for OS-level
access control and disk encryption on the machine holding the database.
"""
from __future__ import annotations

import hashlib
import hmac
import os
import secrets

from .exceptions import AuthError
from .storage.database import Repository

_KEY = "admin_password_hash"
_ITERATIONS = 600_000
ENV_VAR = "FACEATTEND_ADMIN_PASSWORD"


def _hash(password: str, salt: bytes, iterations: int) -> bytes:
    return hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iterations)


class AdminGuard:
    def __init__(self, repo: Repository, enabled: bool = True):
        self.repo = repo
        self.enabled = enabled

    @property
    def is_initialised(self) -> bool:
        return self.repo.get_setting(_KEY) is not None

    def set_password(self, password: str, current: str | None = None) -> None:
        if len(password) < 10:
            raise AuthError("Admin passphrase must be at least 10 characters")
        if self.is_initialised:
            self.verify(current or "")
        salt = secrets.token_bytes(16)
        digest = _hash(password, salt, _ITERATIONS)
        self.repo.set_setting(_KEY, f"pbkdf2_sha256${_ITERATIONS}${salt.hex()}${digest.hex()}")
        self.repo.audit("admin_password_set")

    def verify(self, password: str) -> None:
        stored = self.repo.get_setting(_KEY)
        if stored is None:
            raise AuthError("Admin passphrase not set. Run: faceattend init-admin")
        try:
            algo, iters, salt_hex, hash_hex = stored.split("$")
            assert algo == "pbkdf2_sha256"
        except (ValueError, AssertionError) as exc:
            raise AuthError("Stored admin credential is malformed") from exc
        candidate = _hash(password, bytes.fromhex(salt_hex), int(iters))
        if not hmac.compare_digest(candidate, bytes.fromhex(hash_hex)):
            self.repo.audit("admin_auth_failed")
            raise AuthError("Invalid admin passphrase")

    def require(self, prompt) -> None:
        """Verify using $FACEATTEND_ADMIN_PASSWORD or the interactive ``prompt`` callable."""
        if not self.enabled:
            return
        password = os.environ.get(ENV_VAR)
        if password is None:
            password = prompt()
        self.verify(password)
