"""Smoke tests for Phase 0 package scaffold."""

from rebirth_ai import __version__


def test_version() -> None:
    assert __version__ == "0.1.0"
