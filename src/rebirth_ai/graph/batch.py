"""Multi-file migration orchestration (Phase 5)."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable

from rebirth_ai.adapters.js_ts import JsTsAdapter, SOURCE_SUFFIXES
from rebirth_ai.graph.app import build_migration_graph, initial_state

SKIP_DIR_NAMES = {
    "node_modules",
    ".git",
    "dist",
    "build",
    "coverage",
    ".venv",
    "venv",
    "data",
    "__pycache__",
}


@dataclass
class FileMigrationResult:
    relative_file: str
    relative_test: str | None
    status: str  # success | human_review | failed | skipped_no_test | skipped
    iterations: int = 0
    error_class: str | None = None
    trace: list[str] = field(default_factory=list)
    wrote: bool = False
    message: str = ""


@dataclass
class BatchReport:
    project_dir: str
    objective: str
    results: list[FileMigrationResult] = field(default_factory=list)

    @property
    def succeeded(self) -> list[FileMigrationResult]:
        return [r for r in self.results if r.status == "success"]

    @property
    def failed(self) -> list[FileMigrationResult]:
        return [r for r in self.results if r.status not in {"success", "skipped", "skipped_no_test"}]

    def to_dict(self) -> dict:
        return {
            "project_dir": self.project_dir,
            "objective": self.objective,
            "summary": {
                "total": len(self.results),
                "success": len(self.succeeded),
                "failed": len(self.failed),
                "skipped": sum(
                    1 for r in self.results if r.status.startswith("skipped")
                ),
            },
            "results": [asdict(r) for r in self.results],
        }

    def write_json(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")


def discover_source_files(
    project_dir: Path,
    *,
    adapter: JsTsAdapter | None = None,
    under: str = "src",
) -> list[Path]:
    """Return project-relative source paths under ``under`` (default ``src/``)."""
    adapter = adapter or JsTsAdapter()
    root = project_dir.resolve()
    base = root / under if under else root
    if not base.exists():
        return []
    out: list[Path] = []
    for path in sorted(base.rglob("*")):
        if not path.is_file():
            continue
        if any(part in SKIP_DIR_NAMES for part in path.parts):
            continue
        if path.suffix.lower() not in SOURCE_SUFFIXES:
            continue
        if not adapter.is_source_file(path):
            continue
        out.append(path.relative_to(root))
    return out


def infer_test_path(project_dir: Path, relative_file: Path | str) -> str | None:
    """Map ``src/foo.js`` → ``test/foo.test.js`` when that file exists."""
    rel = Path(relative_file)
    stem = rel.stem
    candidates = [
        Path("test") / f"{stem}.test.js",
        Path("test") / f"{stem}.test.ts",
        Path("tests") / f"{stem}.test.js",
        Path("tests") / f"{stem}.test.ts",
        Path("__tests__") / f"{stem}.test.js",
    ]
    for cand in candidates:
        if (project_dir / cand).is_file():
            return cand.as_posix()
    return None


def _parse_seed_map(raw: list[str] | None) -> dict[str, Path]:
    """Parse ``rel/path.js=/abs/or/rel/seed.js`` entries."""
    mapping: dict[str, Path] = {}
    for item in raw or []:
        if "=" not in item:
            raise ValueError(f"Invalid --seed entry (want file=seed): {item}")
        file_part, seed_part = item.split("=", 1)
        mapping[file_part.strip()] = Path(seed_part.strip())
    return mapping


def migrate_repository(
    project_dir: Path,
    *,
    objective: str,
    files: list[str] | None = None,
    under: str = "src",
    seeds: list[str] | None = None,
    write: bool = False,
    db_path: Path | str = "data/embeddings.db",
    max_iterations: int = 5,
    skip_without_tests: bool = True,
    on_file: Callable[[str], None] | None = None,
) -> BatchReport:
    """Run the single-file LangGraph loop across many sources; continue on failures."""
    project_dir = project_dir.resolve()
    seed_map = _parse_seed_map(seeds)
    if files:
        targets = [Path(f) for f in files]
    else:
        targets = discover_source_files(project_dir, under=under)

    graph = build_migration_graph()
    report = BatchReport(project_dir=str(project_dir), objective=objective)

    for rel in targets:
        rel_s = rel.as_posix()
        if on_file:
            on_file(rel_s)
        test_rel = infer_test_path(project_dir, rel)
        if test_rel is None and skip_without_tests:
            report.results.append(
                FileMigrationResult(
                    relative_file=rel_s,
                    relative_test=None,
                    status="skipped_no_test",
                    message="No matching test file; skipped",
                )
            )
            continue

        seed_path = seed_map.get(rel_s)
        seed_text = None
        if seed_path is not None:
            seed_path = seed_path if seed_path.is_absolute() else Path.cwd() / seed_path
            seed_text = seed_path.read_text(encoding="utf-8")

        state = initial_state(
            project_dir=project_dir,
            relative_file=rel_s,
            relative_test=test_rel or "",
            objective=objective,
            seed_draft=seed_text,
            test_first=seed_text is not None,
            db_path=db_path,
            max_iterations=max_iterations,
        )

        try:
            final = graph.invoke(state)
        except Exception as exc:  # noqa: BLE001
            report.results.append(
                FileMigrationResult(
                    relative_file=rel_s,
                    relative_test=test_rel,
                    status="failed",
                    message=str(exc),
                )
            )
            continue

        status = str(final.get("status") or "failed")
        draft = final.get("current_draft") or ""
        wrote = False
        if write and status == "success" and draft:
            target = project_dir / rel_s
            target.write_text(draft, encoding="utf-8")
            wrote = True

        report.results.append(
            FileMigrationResult(
                relative_file=rel_s,
                relative_test=test_rel,
                status=status,
                iterations=int(final.get("iteration_count") or 0),
                error_class=final.get("error_class"),
                trace=list(final.get("trace") or []),
                wrote=wrote,
            )
        )

    return report


def render_pr_body(report: BatchReport) -> str:
    """Markdown PR summary: successes + failed files (do not abort whole run)."""
    lines = [
        "## Rebirth AI migration report",
        "",
        f"**Project:** `{report.project_dir}`",
        f"**Objective:** {report.objective}",
        "",
        "### Summary",
        "",
        f"- Success: **{len(report.succeeded)}**",
        f"- Failed / human review: **{len(report.failed)}**",
        f"- Skipped: **{sum(1 for r in report.results if r.status.startswith('skipped'))}**",
        "",
        "### Succeeded",
        "",
    ]
    if report.succeeded:
        for r in report.succeeded:
            lines.append(
                f"- `{r.relative_file}` (iters={r.iterations}"
                f"{', wrote' if r.wrote else ''})"
            )
    else:
        lines.append("- _(none)_")

    lines.extend(["", "### Failed / needs review", ""])
    if report.failed:
        for r in report.failed:
            err = r.error_class or r.message or r.status
            lines.append(f"- `{r.relative_file}` — `{err}`")
    else:
        lines.append("- _(none)_")

    lines.extend(["", "---", "_Generated by `rebirth migrate-repo`._", ""])
    return "\n".join(lines)
