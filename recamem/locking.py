"""Single-writer lock per profile.

Sleep, teaching and any other writer take an exclusive advisory lock on the
profile directory. A second writer fails with a named error instead of waiting
indefinitely (memory/sleep-consolidation, knowledge-store).
"""

from __future__ import annotations

import os
from pathlib import Path
from types import TracebackType
from typing import Optional

try:  # POSIX advisory locking
    import fcntl
except ImportError:  # pragma: no cover - Windows is not a supported writer target
    fcntl = None  # type: ignore[assignment]

LOCK_NAME = ".recamem-writer.lock"


class ProfileBusyError(RuntimeError):
    """Another writer holds the profile's writer lock."""

    def __init__(self, profile: Path, holder: Optional[str] = None):
        self.profile = Path(profile)
        self.holder = holder
        detail = f" (held by {holder})" if holder else ""
        super().__init__(
            f"Profile {self.profile} already has a writer{detail}; "
            "recamem runs one writer per profile"
        )


class ProfileWriterLock:
    """Exclusive, non-blocking writer lock for one profile directory."""

    def __init__(self, profile_dir: os.PathLike | str):
        self.profile_dir = Path(profile_dir)
        self.path = self.profile_dir / LOCK_NAME
        self._fd: Optional[int] = None

    def acquire(self) -> "ProfileWriterLock":
        self.profile_dir.mkdir(parents=True, exist_ok=True)
        if fcntl is None:
            # Best effort: an in-process marker still refuses a second writer.
            if self.path.exists():
                raise ProfileBusyError(self.profile_dir, self.path.read_text().strip() or None)
            self.path.write_text(str(os.getpid()))
            self._fd = -1
            return self
        fd = os.open(self.path, os.O_RDWR | os.O_CREAT, 0o644)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            try:
                holder = os.read(fd, 64).decode(errors="replace").strip() or None
            except OSError:
                holder = None
            os.close(fd)
            raise ProfileBusyError(self.profile_dir, holder)
        os.ftruncate(fd, 0)
        os.write(fd, str(os.getpid()).encode())
        os.fsync(fd)
        self._fd = fd
        return self

    def release(self) -> None:
        if self._fd is None:
            return
        if self._fd == -1:
            try:
                self.path.unlink()
            except FileNotFoundError:
                pass
            self._fd = None
            return
        try:
            fcntl.flock(self._fd, fcntl.LOCK_UN)
        finally:
            os.close(self._fd)
            self._fd = None

    @property
    def held(self) -> bool:
        return self._fd is not None

    def __enter__(self) -> "ProfileWriterLock":
        return self.acquire()

    def __exit__(self, exc_type, exc, tb) -> None:
        self.release()
