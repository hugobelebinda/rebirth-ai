"""Multi-file migration orchestration (Phases 5–6)."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path

from rebirth_ai.adapters.registry import get_adapter_for_project
from rebirth_ai.graph.app import build_migration_graph, initial_state
from rebirth_ai.graph.discovery import (
    DENY_DIR_NAMES,
    discover_source_files,
    extract_local_imports,
    infer_test_path,
    topological_migrate_order,
)
from rebirth_ai.graph.fallback import needs_fallback, run_fallback
from rebirth_ai.paths import DEFAULT_DB
from rebirth_ai.safety import (
    file_too_large,
    has_restrictive_license_header,
    is_secret_path,
    max_file_bytes,
)
from rebirth_ai.textfile import write_like

# Re-export for older imports / tests
SKIP_DIR_NAMES = DENY_DIR_NAMES
__all__ = [
    "BatchReport",
    "FileMigrationResult",
    "SKIP_DIR_NAMES",
    "discover_source_files",
    "infer_test_path",
    "migrate_repository",
    "render_pr_body",
    "topological_migrate_order",
]


@dataclass
class FileMigrationResult:
    relative_file: str
    relative_test: str | None
    status: str  # success | unchanged | human_review | failed | skipped_* | planned
    iterations: int = 0
    error_class: str | None = None
    trace: list[str] = field(default_factory=list)
    wrote: bool = False
    message: str = ""
    test_logs_excerpt: str = ""
    draft_preview: str = ""
    fallback_model: str | None = None  # the 14B that re-ran a file the served adapter left unchanged


@dataclass
class BatchReport:
    project_dir: str
    objective: str
    results: list[FileMigrationResult] = field(default_factory=list)
    dry_run: bool = False
    migrate_order: list[str] = field(default_factory=list)

    @property
    def succeeded(self) -> list[FileMigrationResult]:
        return [r for r in self.results if r.status == "success"]

    @property
    def unchanged(self) -> list[FileMigrationResult]:
        return [r for r in self.results if r.status == "unchanged"]

    @property
    def failed(self) -> list[FileMigrationResult]:
        return [
            r
            for r in self.results
            if r.status
            not in {
                "success",
                "unchanged",
                "skipped",
                "skipped_no_test",
                "skipped_wrong_language",
                "skipped_secret",
                "skipped_too_large",
                "skipped_license",
                "planned",
            }
        ]

    def to_dict(self) -> dict:
        return {
            "project_dir": self.project_dir,
            "objective": self.objective,
            "dry_run": self.dry_run,
            "migrate_order": list(self.migrate_order),
            "summary": {
                "total": len(self.results),
                "success": len(self.succeeded),
                "unchanged": len(self.unchanged),
                "failed": len(self.failed),
                "skipped": sum(
                    1 for r in self.results if r.status.startswith("skipped")
                ),
                "planned": sum(1 for r in self.results if r.status == "planned"),
            },
            "results": [asdict(r) for r in self.results],
        }

    def write_json(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")


UNCHANGED_MESSAGE = "tests pass, but the model returned the file unchanged (at most whitespace differs); nothing written"


def no_change(draft: str, original: str, seed: str | None = None) -> bool:
    """True when a passing draft leaves the file as it was: same code as the original up to
    whitespace, and no seed draft that differed from it (a repaired seed is a real change)."""
    same = draft.split() == original.split()
    return same and (seed is None or seed.split() == original.split())


def _parse_seed_map(raw: list[str] | None) -> dict[str, Path]:
    """Parse ``rel/path.js=/abs/or/rel/seed.js`` entries."""
    mapping: dict[str, Path] = {}
    for item in raw or []:
        if "=" not in item:
            raise ValueError(f"Invalid --seed entry (want file=seed): {item}")
        file_part, seed_part = item.split("=", 1)
        mapping[file_part.strip()] = Path(seed_part.strip())
    return mapping


def _excerpt(text: str | None, *, limit: int = 600) -> str:
    if not text:
        return ""
    cleaned = text.strip()
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[:limit] + "\n...[truncated]"


def _unified_diff(path: str, before: str, after: str, *, max_lines: int = 80) -> str:
    import difflib

    diff = difflib.unified_diff(
        before.splitlines(),
        after.splitlines(),
        fromfile=f"a/{path}",
        tofile=f"b/{path}",
        lineterm="",
    )
    lines = list(diff)
    if len(lines) > max_lines:
        lines = lines[:max_lines] + [f"...[{len(lines) - max_lines} more diff lines truncated]"]
    return "\n".join(lines)


def _file_result(final: dict, project_dir: Path, rel_s: str, test_rel: str | None, original: str,
                 seed_text: str | None, write: bool) -> FileMigrationResult:
    """Report entry for one finished run (writes the draft when it passed and changed the file)."""
    status = str(final.get("status") or "failed")
    draft = final.get("current_draft") or ""
    message = ""
    if status == "success" and draft and no_change(draft, original, seed_text):
        status, message = "unchanged", UNCHANGED_MESSAGE
    wrote = False
    if write and status == "success" and draft:
        write_like(project_dir / rel_s, draft)
        wrote = True
    return FileMigrationResult(
        relative_file=rel_s,
        relative_test=test_rel,
        status=status,
        iterations=int(final.get("iteration_count") or 0),
        error_class=final.get("error_class"),
        trace=list(final.get("trace") or []),
        wrote=wrote,
        message=message,
        test_logs_excerpt=_excerpt(final.get("test_logs")),
        draft_preview=_unified_diff(rel_s, original, draft) if draft and draft != original else "",
        fallback_model=final.get("fallback_model"),
    )


def migrate_repository(
    project_dir: Path,
    *,
    objective: str,
    files: list[str] | None = None,
    under: str | list[str] = "src",
    seeds: list[str] | None = None,
    write: bool = False,
    dry_run: bool = False,
    db_path: Path | str = DEFAULT_DB,
    max_iterations: int = 5,
    skip_without_tests: bool = True,
    lang: str | None = None,
    on_file: Callable[[str], None] | None = None,
    fallback: bool | None = None,
) -> BatchReport:
    """Run the single-file LangGraph loop across many sources; continue on failures.

    ``dry_run=True`` discovers + topo-orders targets and reports the plan without
    invoking the LLM / sandbox or writing files.
    """
    project_dir = project_dir.resolve()
    adapter = get_adapter_for_project(project_dir, lang=lang)
    seed_map = _parse_seed_map(seeds)
    if files:
        targets = [Path(f) for f in files]
    else:
        targets = discover_source_files(project_dir, adapter=adapter, under=under)

    targets = topological_migrate_order(project_dir, targets)
    order = [t.as_posix() for t in targets]
    report = BatchReport(
        project_dir=str(project_dir),
        objective=objective,
        dry_run=dry_run,
        migrate_order=order,
    )

    def _precheck(rel: Path) -> FileMigrationResult | None:
        """Return a skip result, or None if the file may proceed."""
        rel_s = rel.as_posix()
        abs_path = project_dir / rel_s
        # Secret check first: a credential path must be reported as such even when
        # the adapter would not own it anyway (truthful report / PR body).
        if is_secret_path(rel):
            return FileMigrationResult(
                relative_file=rel_s,
                relative_test=None,
                status="skipped_secret",
                message="Secret/credential path blocked",
            )
        if not adapter.is_source_file(abs_path):
            return FileMigrationResult(
                relative_file=rel_s,
                relative_test=None,
                status="skipped_wrong_language",
                message=f"Not owned by adapter {adapter.name}",
            )
        if file_too_large(abs_path):
            return FileMigrationResult(
                relative_file=rel_s,
                relative_test=None,
                status="skipped_too_large",
                message=f"File exceeds REBIRTH_MAX_FILE_BYTES={max_file_bytes()}",
            )
        try:
            text = abs_path.read_text(encoding="utf-8")
        except OSError as exc:
            return FileMigrationResult(
                relative_file=rel_s,
                relative_test=None,
                status="failed",
                message=f"read failed: {exc}",
            )
        if has_restrictive_license_header(text):
            return FileMigrationResult(
                relative_file=rel_s,
                relative_test=infer_test_path(project_dir, rel),
                status="skipped_license",
                message="Restrictive license header; skipped",
            )
        return None

    if dry_run:
        for rel in targets:
            rel_s = rel.as_posix()
            if on_file:
                on_file(rel_s)
            blocked = _precheck(rel)
            if blocked is not None:
                report.results.append(blocked)
                continue
            test_rel = infer_test_path(project_dir, rel)
            deps = extract_local_imports(project_dir, rel)
            if test_rel is None and skip_without_tests:
                report.results.append(
                    FileMigrationResult(
                        relative_file=rel_s,
                        relative_test=None,
                        status="skipped_no_test",
                        message="No matching test file; would skip",
                    )
                )
                continue
            dep_note = f"deps={deps}" if deps else "deps=[]"
            report.results.append(
                FileMigrationResult(
                    relative_file=rel_s,
                    relative_test=test_rel,
                    status="planned",
                    message=f"dry-run plan; {dep_note}",
                )
            )
        return report

    graph = build_migration_graph()
    deferred: list[tuple] = []

    for rel in targets:
        rel_s = rel.as_posix()
        if on_file:
            on_file(rel_s)
        blocked = _precheck(rel)
        if blocked is not None:
            report.results.append(blocked)
            continue
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

        original = (project_dir / rel_s).read_text(encoding="utf-8")
        state = initial_state(
            project_dir=project_dir,
            relative_file=rel_s,
            relative_test=test_rel or "",
            objective=objective,
            seed_draft=seed_text,
            test_first=seed_text is not None,
            db_path=db_path,
            max_iterations=max_iterations,
            adapter_id=adapter.name,
        )

        try:
            final = graph.invoke(state)
        except Exception as exc:
            report.results.append(
                FileMigrationResult(
                    relative_file=rel_s,
                    relative_test=test_rel,
                    status="failed",
                    message=str(exc),
                )
            )
            continue
        report.results.append(_file_result(final, project_dir, rel_s, test_rel, original, seed_text, write))
        served = needs_fallback(final, original=original, seed=seed_text, use=fallback)
        if served:
            deferred.append((len(report.results) - 1, state, final, served, rel_s, test_rel, original, seed_text))

    # Fallback runs last: the served 7B and the 14B do not fit in a 16 GB GPU together, so one pass
    # per model swaps once instead of once per unchanged file.
    for idx, state, final, served, rel_s, test_rel, original, seed_text in deferred:
        if on_file:
            on_file(f"{rel_s} (fallback)")
        try:
            final = run_fallback(graph, state, final, served)
        except Exception as exc:  # keep the 7B result, note why the fallback did not run
            report.results[idx].trace.append(f"fallback: failed — {exc}")
            continue
        report.results[idx] = _file_result(final, project_dir, rel_s, test_rel, original, seed_text, write)

    return report


def render_pr_body(report: BatchReport, *, project_label: str | None = None) -> str:
    """Markdown PR summary: successes, failures, optional sandbox excerpts. ``project_label``
    replaces the project's absolute path (a PR should not show local paths)."""
    lines = [
        "## Rebirth AI migration report",
        "",
        f"**Project:** `{project_label or report.project_dir}`",
        f"**Objective:** {report.objective}",
    ]
    if report.dry_run:
        lines.append("**Mode:** dry-run (plan only; no LLM / writes)")
    if report.migrate_order:
        lines.extend(
            [
                "",
                "### Migrate order (dependency-aware)",
                "",
            ]
        )
        for i, path in enumerate(report.migrate_order, 1):
            lines.append(f"{i}. `{path}`")

    lines.extend(
        [
            "",
            "### Summary",
            "",
            f"- Success: **{len(report.succeeded)}**",
            f"- Unchanged (tests pass, the model changed nothing): **{len(report.unchanged)}**",
            f"- Failed / human review: **{len(report.failed)}**",
            f"- Skipped: **{sum(1 for r in report.results if r.status.startswith('skipped'))}**",
            f"- Planned: **{sum(1 for r in report.results if r.status == 'planned')}**",
            "",
            "### Succeeded",
            "",
        ]
    )
    if report.succeeded:
        for r in report.succeeded:
            lines.append(
                f"- `{r.relative_file}` (iters={r.iterations}"
                f"{', wrote' if r.wrote else ''}"
                f"{f', by the fallback {r.fallback_model}' if r.fallback_model else ''})"
            )
    else:
        lines.append("- _(none)_")

    if report.unchanged:
        lines.extend(["", "### Unchanged", ""])
        for r in report.unchanged:
            lines.append(f"- `{r.relative_file}` — {r.message}")

    lines.extend(["", "### Failed / needs review", ""])
    if report.failed:
        for r in report.failed:
            err = r.error_class or r.message or r.status
            lines.append(f"- `{r.relative_file}` — `{err}`")
            if r.test_logs_excerpt:
                lines.append("")
                lines.append("  <details><summary>Sandbox log excerpt</summary>")
                lines.append("")
                lines.append("  ```text")
                for log_line in r.test_logs_excerpt.splitlines():
                    lines.append(f"  {log_line}")
                lines.append("  ```")
                lines.append("")
                lines.append("  </details>")
    else:
        lines.append("- _(none)_")

    planned = [r for r in report.results if r.status == "planned"]
    if planned:
        lines.extend(["", "### Planned (dry-run)", ""])
        for r in planned:
            test = r.relative_test or "_(no test)_"
            lines.append(f"- `{r.relative_file}` → test `{test}` — {r.message}")

    lines.extend([
        "",
        "> Passing tests are evidence, not proof: code paths the tests do not run can still change "
        "behavior. Review every diff before merging.",
        "",
        "---",
        "_Generated by `rebirth migrate-repo`._",
        "",
    ])
    return "\n".join(lines)
