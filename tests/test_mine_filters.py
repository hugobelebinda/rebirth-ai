"""Unit tests for migration-pair mining filters (no Hub download)."""

from __future__ import annotations

from training.mine_migration_pairs import (
    _looks_like_migration,
    _quality_ok,
    dedupe,
    Pair,
)


def test_migration_keywords_accept() -> None:
    assert _looks_like_migration("Migrate to ESM imports", "")
    assert _looks_like_migration("Upgrade react 17 to 18", "")
    assert _looks_like_migration("Replace var with const", "")


def test_migration_keywords_deny() -> None:
    assert not _looks_like_migration("fix typo in README", "")
    assert not _looks_like_migration("Add new feature for checkout", "")


def test_quality_bounds() -> None:
    assert _quality_ok("a" * 50, "b" * 50, min_chars=40, max_chars=100)
    assert not _quality_ok("same", "same", min_chars=1, max_chars=100)
    assert not _quality_ok("x", "y", min_chars=40, max_chars=100)


def test_dedupe() -> None:
    a = Pair("o", "javascript", "legacy", "modern", "synthetic")
    b = Pair("o2", "javascript", "legacy", "modern", "commitpackft")
    c = Pair("o3", "javascript", "other", "modern2", "commitpackft")
    assert len(dedupe([a, b, c])) == 2
