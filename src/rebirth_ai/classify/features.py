"""Feature helpers for error-log classification."""

from __future__ import annotations

ERROR_CLASSES = (
    "SyntaxError",
    "DependencyMissing",
    "TypeMismatch",
    "TestLogicFailure",
)


def normalize_log(text: str) -> str:
    """Light cleanup so TF-IDF focuses on tokens that matter."""
    return " ".join(text.replace("\t", " ").split())
