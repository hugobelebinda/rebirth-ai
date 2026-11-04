"""Phase 7 safety filters."""

from __future__ import annotations

from pathlib import Path

from rebirth_ai.graph.batch import migrate_repository
from rebirth_ai.safety import (
    filter_secret_paths,
    has_restrictive_license_header,
    is_secret_path,
)


def test_secret_path_detection() -> None:
    assert is_secret_path(".env")
    assert is_secret_path("config/.env.local")
    assert is_secret_path("certs/server.pem")
    assert is_secret_path("secrets/token.json")
    assert not is_secret_path("src/math.js")


def test_filter_secret_paths() -> None:
    safe, blocked = filter_secret_paths(["src/a.js", ".env", "lib/b.ts"])
    assert safe == ["src/a.js", "lib/b.ts"]
    assert blocked == [".env"]


def test_restrictive_license_header() -> None:
    assert has_restrictive_license_header(
        "/* Proprietary and confidential. All rights reserved. */\nexport const x = 1;\n"
    )
    assert not has_restrictive_license_header("/* MIT */\nexport const x = 1;\n")


def test_migrate_skips_secret_and_large(tmp_path: Path, monkeypatch) -> None:
    src = tmp_path / "src"
    src.mkdir()
    (src / "ok.js").write_text("export const a = 1;\n", encoding="utf-8")
    (src / ".env").write_text("SECRET=1\n", encoding="utf-8")
    monkeypatch.setenv("REBIRTH_MAX_FILE_BYTES", "32")
    (src / "huge.js").write_text("x" * 100, encoding="utf-8")

    report = migrate_repository(
        tmp_path,
        objective="demo",
        files=["src/.env", "src/huge.js", "src/ok.js"],
        dry_run=True,
        skip_without_tests=False,
    )
    statuses = {r.relative_file: r.status for r in report.results}
    assert statuses["src/.env"] == "skipped_secret"
    assert statuses["src/huge.js"] == "skipped_too_large"
    assert statuses["src/ok.js"] == "planned"
