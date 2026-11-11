"""Product safety filters for migrate-repo / PR staging (Phase 7)."""

from __future__ import annotations

import os
import re
from pathlib import Path

# Path basename / relative patterns that must never be committed or migrated as source
SECRET_BASENAMES = frozenset(
    {
        ".env",
        ".env.local",
        ".env.production",
        ".env.development",
        "credentials.json",
        "service-account.json",
        "id_rsa",
        "id_ed25519",
        "secrets.json",
    }
)

SECRET_SUFFIXES = frozenset({".pem", ".p12", ".pfx", ".key"})

SECRET_PATH_GLOBS = (
    re.compile(r"(^|/)\.env(\.|$)"),
    re.compile(r"(^|/)secrets?(/|$)"),
    re.compile(r"(^|/)credentials?(/|$)"),
)

# Optional: skip files whose header mentions restrictive licenses
RESTRICTIVE_LICENSE_MARKERS = (
    "all rights reserved",
    "proprietary and confidential",
    "not for redistribution",
)

DEFAULT_MAX_FILE_BYTES = int(os.environ.get("REBIRTH_MAX_FILE_BYTES", str(200 * 1024)))


def is_secret_path(relative: Path | str) -> bool:
    """True if path looks like a secret / credential file."""
    rel = Path(relative)
    name = rel.name
    if name in SECRET_BASENAMES:
        return True
    if name.startswith(".env."):
        return True
    if rel.suffix.lower() in SECRET_SUFFIXES:
        return True
    posix = rel.as_posix()
    return any(rx.search(posix) for rx in SECRET_PATH_GLOBS)


def max_file_bytes() -> int:
    return int(os.environ.get("REBIRTH_MAX_FILE_BYTES", str(DEFAULT_MAX_FILE_BYTES)))


def file_too_large(path: Path, *, limit: int | None = None) -> bool:
    cap = limit if limit is not None else max_file_bytes()
    try:
        return path.is_file() and path.stat().st_size > cap
    except OSError:
        return False


def has_restrictive_license_header(text: str, *, max_chars: int = 800) -> bool:
    head = text[:max_chars].lower()
    return any(marker in head for marker in RESTRICTIVE_LICENSE_MARKERS)


def filter_secret_paths(paths: list[str]) -> tuple[list[str], list[str]]:
    """Split paths into (safe_to_stage, blocked_secrets)."""
    safe: list[str] = []
    blocked: list[str] = []
    for p in paths:
        if is_secret_path(p):
            blocked.append(p)
        else:
            safe.append(p)
    return safe, blocked


class InputError(ValueError):
    """A path or file Rebirth refuses to work on; the CLI prints it without a traceback."""


def inside(root: Path | str, relative: Path | str, *, what: str = "the project") -> Path:
    """``root / relative`` resolved (symlinks followed); refuses anything that lands outside ``root``."""
    base = Path(root).resolve()
    path = (base / relative).resolve()
    try:
        path.relative_to(base)
    except ValueError:
        raise InputError(f"{relative} is outside {what} ({base})") from None
    return path


def check_source_file(project: Path | str, relative: Path | str) -> str:
    """The text of a file Rebirth is asked to migrate or translate, after checking it is inside the
    project, exists, is not a secrets file, is not too large, and is non-empty UTF-8 text."""
    path = inside(project, relative)
    if not path.exists():
        raise InputError(f"no such file: {relative} (in {Path(project).resolve()})")
    if not path.is_file():
        raise InputError(f"{relative} is not a file")
    if is_secret_path(path.relative_to(Path(project).resolve())):
        raise InputError(f"{relative} looks like a secrets file: Rebirth does not send it to a model or rewrite it")
    if file_too_large(path):
        raise InputError(f"{relative} is {path.stat().st_size:,} bytes, over REBIRTH_MAX_FILE_BYTES={max_file_bytes():,}")
    data = path.read_bytes()
    if b"\x00" in data[:8192]:
        raise InputError(f"{relative} is a binary file")
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        raise InputError(f"{relative} is not UTF-8 text; convert it first (e.g. iconv -f latin1 -t utf-8)") from None
    if not text.strip():
        raise InputError(f"{relative} is empty: nothing to migrate")
    return text
