"""Phase 6 discovery: multi-root, gitignore, topo-order, test mapping."""

from __future__ import annotations

from pathlib import Path

from rebirth_ai.graph.batch import migrate_repository, render_pr_body
from rebirth_ai.graph.discovery import (
    discover_source_files,
    extract_local_imports,
    infer_test_path,
    is_ignored_path,
    parse_under_roots,
    topological_migrate_order,
)

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "js_ts_sample"


def test_parse_under_roots_csv() -> None:
    assert parse_under_roots("src,lib,app") == ["src", "lib", "app"]
    assert parse_under_roots(" src ") == ["src"]


def test_topo_order_format_before_math() -> None:
    files = [Path("src/math.js"), Path("src/format.js")]
    ordered = topological_migrate_order(FIXTURE, files)
    assert [p.as_posix() for p in ordered] == ["src/format.js", "src/math.js"]


def test_extract_local_imports_math() -> None:
    deps = extract_local_imports(FIXTURE, "src/math.js")
    assert "src/format.js" in deps


def test_infer_spec_and_colocated_patterns(tmp_path: Path) -> None:
    src = tmp_path / "src"
    src.mkdir()
    (src / "util.js").write_text("export const x = 1;\n", encoding="utf-8")
    (tmp_path / "test").mkdir()
    (tmp_path / "test" / "util.spec.js").write_text("import test from 'node:test';\n", encoding="utf-8")
    assert infer_test_path(tmp_path, "src/util.js") == "test/util.spec.js"

    (src / "__tests__").mkdir()
    (src / "other.js").write_text("export const y = 2;\n", encoding="utf-8")
    (src / "__tests__" / "other.test.js").write_text("//\n", encoding="utf-8")
    assert infer_test_path(tmp_path, "src/other.js") == "src/__tests__/other.test.js"


def test_gitignore_and_deny_list(tmp_path: Path) -> None:
    (tmp_path / ".gitignore").write_text("secret.js\nignored_dir/\n", encoding="utf-8")
    src = tmp_path / "src"
    src.mkdir()
    (src / "ok.js").write_text("export const a = 1;\n", encoding="utf-8")
    (src / "secret.js").write_text("export const b = 2;\n", encoding="utf-8")
    ignored = src / "ignored_dir"
    ignored.mkdir()
    (ignored / "nope.js").write_text("export const c = 3;\n", encoding="utf-8")
    nm = src / "node_modules" / "pkg"
    nm.mkdir(parents=True)
    (nm / "index.js").write_text("module.exports = 1;\n", encoding="utf-8")

    found = {p.as_posix() for p in discover_source_files(tmp_path, under="src")}
    assert "src/ok.js" in found
    assert "src/secret.js" not in found
    assert "src/ignored_dir/nope.js" not in found
    assert not any("node_modules" in p for p in found)
    assert is_ignored_path(tmp_path, "src/node_modules/pkg/index.js")


def test_multi_root_under(tmp_path: Path) -> None:
    for root_name in ("src", "lib"):
        d = tmp_path / root_name
        d.mkdir()
        (d / f"{root_name}_file.js").write_text("export const x = 1;\n", encoding="utf-8")
    found = {p.as_posix() for p in discover_source_files(tmp_path, under="src,lib")}
    assert "src/src_file.js" in found
    assert "lib/lib_file.js" in found


def test_monorepo_package_src_to_test_mapping() -> None:
    mono = Path(__file__).resolve().parents[1] / "fixtures" / "bench_monorepo"
    assert (
        infer_test_path(mono, "packages/a/src/hello.js")
        == "packages/a/test/hello.test.js"
    )
    files = discover_source_files(mono, under="packages/a,packages/b")
    names = {p.as_posix() for p in files}
    assert "packages/a/src/hello.js" in names
    assert "packages/b/src/world.js" in names
    assert not any("test/" in n or n.endswith(".test.js") for n in names)


def test_dry_run_migrate_repo_fixture() -> None:
    report = migrate_repository(
        FIXTURE,
        objective="dry",
        under="src",
        dry_run=True,
        skip_without_tests=True,
    )
    assert report.dry_run is True
    assert report.migrate_order[0] == "src/format.js"
    assert "src/math.js" in report.migrate_order
    statuses = {r.relative_file: r.status for r in report.results}
    assert statuses["src/format.js"] == "planned"
    assert statuses["src/math.js"] == "planned"
    assert statuses.get("src/greeter.ts") == "skipped_no_test"
    body = render_pr_body(report)
    assert "dry-run" in body
    assert "Migrate order" in body


def test_infer_falls_back_to_a_test_that_imports_the_file(tmp_path: Path) -> None:
    from rebirth_ai.adapters.js_ts import JsTsAdapter

    # single-file npm package: root test.js requires the package root (package.json "main")
    (tmp_path / "index.js").write_text("module.exports = () => 1;\n", encoding="utf-8")
    (tmp_path / "test.js").write_text("var f = require('./');\n", encoding="utf-8")
    (tmp_path / "package.json").write_text('{"name": "pkg", "main": "index.js"}', encoding="utf-8")
    assert infer_test_path(tmp_path, "index.js") == "test.js"  # no test script: node --test test.js
    (tmp_path / "package.json").write_text(
        '{"name": "pkg", "main": "index.js", "scripts": {"test": "mocha"}}', encoding="utf-8")
    assert infer_test_path(tmp_path, "index.js") == "package.json"  # the project's own runner
    assert JsTsAdapter().test_command_for("package.json") == "npm test"
    # a file no test imports stays without a test (skipped), even with a test script
    (tmp_path / "other.js").write_text("module.exports = 2;\n", encoding="utf-8")
    assert infer_test_path(tmp_path, "other.js") is None
    # test/ dir importing ../lib/x and node_modules never scanned
    (tmp_path / "lib").mkdir()
    (tmp_path / "lib" / "x.js").write_text("export const x = 1;\n", encoding="utf-8")
    (tmp_path / "test").mkdir()
    (tmp_path / "test" / "x.js").write_text("import { x } from '../lib/x.js';\n", encoding="utf-8")
    nm = tmp_path / "node_modules" / "dep"
    nm.mkdir(parents=True)
    (nm / "test.js").write_text("require('../../other');\n", encoding="utf-8")
    assert infer_test_path(tmp_path, "lib/x.js") == "package.json"
    assert infer_test_path(tmp_path, "other.js") is None


def test_infer_python_package_and_submodule(tmp_path: Path) -> None:
    pkg = tmp_path / "inflect"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("def f():\n    return 1\n", encoding="utf-8")
    (pkg / "util.py").write_text("X = 1\n", encoding="utf-8")
    (tmp_path / "test_inflect.py").write_text("import pytest\n\nimport inflect\n", encoding="utf-8")
    assert infer_test_path(tmp_path, "inflect/__init__.py") == "test_inflect.py"
    assert infer_test_path(tmp_path, "inflect/util.py") is None  # nothing imports inflect.util
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_misc.py").write_text("from inflect import other, util\n", encoding="utf-8")
    assert infer_test_path(tmp_path, "inflect/util.py") == "tests/test_misc.py"


def test_infer_python_reexported_module_runs_the_test_folder(tmp_path: Path) -> None:
    pkg = tmp_path / "strutils"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("from .validation import *\nfrom .generation import *\n", encoding="utf-8")
    (pkg / "validation.py").write_text("def is_x(s):\n    return True\n", encoding="utf-8")
    (pkg / "generation.py").write_text("def gen():\n    return 'x'\n", encoding="utf-8")
    (pkg / "unused.py").write_text("Y = 1\n", encoding="utf-8")
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_is_x.py").write_text("from strutils import is_x\n", encoding="utf-8")
    (tmp_path / "tests" / "test_gen.py").write_text("from strutils import gen\n", encoding="utf-8")
    assert infer_test_path(tmp_path, "strutils/validation.py") == "tests"  # through the package
    assert infer_test_path(tmp_path, "strutils/unused.py") is None  # not re-exported, not imported
    assert infer_test_path(tmp_path, "strutils/__init__.py") == "tests"  # two tests import it
    (tmp_path / "tests" / "test_val.py").write_text("import strutils.validation\n", encoding="utf-8")
    assert infer_test_path(tmp_path, "strutils/validation.py") == "tests/test_val.py"  # its own test first


def test_detect_language_without_marker_files(tmp_path: Path) -> None:
    from rebirth_ai.adapters.go_lang import GoAdapter, module_path
    from rebirth_ai.adapters.registry import get_adapter_for_project

    (tmp_path / "errors.go").write_text("package errors\n", encoding="utf-8")
    (tmp_path / "errors_test.go").write_text("package errors\n", encoding="utf-8")
    (tmp_path / "README.md").write_text("# x\n", encoding="utf-8")
    assert get_adapter_for_project(tmp_path).name == "go"  # Go from before modules: no go.mod
    assert module_path(tmp_path) == "rebirth.local/project"
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "config").write_text(
        '[remote "origin"]\n\turl = https://github.com/pkg/errors.git\n', encoding="utf-8")
    assert module_path(tmp_path) == "github.com/pkg/errors"
    assert GoAdapter().setup_command_for(tmp_path) == (
        "go mod init github.com/pkg/errors && go mod edit -go=1.22 && go mod tidy")
    (tmp_path / "go.mod").write_text("module github.com/pkg/errors\n", encoding="utf-8")
    assert GoAdapter().setup_command_for(tmp_path) is None  # no requires: nothing to download
