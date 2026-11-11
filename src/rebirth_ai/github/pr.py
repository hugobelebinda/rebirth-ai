"""Create a branch, commit migration successes, and open a GitHub PR."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from rebirth_ai.github import client as gh
from rebirth_ai.github.client import GitHubError
from rebirth_ai.graph.batch import BatchReport, render_pr_body
from rebirth_ai.redact import redact_text
from rebirth_ai.safety import filter_secret_paths


@dataclass(frozen=True)
class PullRequestResult:
    branch: str
    base: str
    url: str | None
    committed: bool
    pushed: bool
    body: str
    message: str = ""


def open_migration_pr(
    project: Path,
    report: BatchReport,
    *,
    branch: str = "rebirth/migrate",
    base: str | None = None,
    title: str | None = None,
    paths: list[str] | None = None,
    push: bool = True,
    create_pr: bool = True,
    draft: bool = False,
    report_path: Path | None = None,
) -> PullRequestResult:
    """Commit the files the migration wrote on a new branch of ``project``'s git repo, push it, and
    open a PR.

    ``paths`` are project-relative; the default is every file a successful run wrote. Only these
    files go into the commit: anything else you had staged stays staged, out of the commit. The PR
    body lists failed files too, and shows the project relative to the repo root, with no local
    paths. ``report_path`` is added only when it lies inside the repo. ``base=None`` picks the
    repo's default branch.
    """
    project = Path(project).resolve()
    repo = gh.git_root(project)
    try:
        prefix = project.relative_to(repo.resolve())
    except ValueError:
        prefix = Path(".")
    label = repo.name if str(prefix) == "." else f"{repo.name}/{prefix.as_posix()}"
    body = redact_text(render_pr_body(report, project_label=label))
    base_branch = base or gh.detect_default_branch(repo)
    pr_title = title or f"Rebirth AI: migrate {len(report.succeeded)} file(s)"

    if paths is None:
        paths = [r.relative_file for r in report.succeeded if r.wrote]
    rel = [(prefix / p).as_posix() for p in paths]
    if report_path is not None:
        try:
            rel.append(Path(report_path).resolve().relative_to(repo.resolve()).as_posix())
        except ValueError:
            pass  # outside the repo: the PR body has the summary
    rel, blocked = filter_secret_paths(rel)
    if blocked:
        raise GitHubError("Refusing to stage secret/credential paths in migration PR: " + ", ".join(blocked))
    if not rel:
        return PullRequestResult(branch=branch, base=base_branch, url=None, committed=False, pushed=False,
                                 body=body, message="Nothing to commit: no file was written (did every file fail?)")

    gh.create_branch(repo, branch)
    gh.stage_paths(repo, rel)
    committed = gh.commit(
        repo,
        f"rebirth: migrate {len(report.succeeded)} file(s)\n\nFailed/review: {len(report.failed)}",
        rel,
    )
    if not committed:
        return PullRequestResult(branch=branch, base=base_branch, url=None, committed=False, pushed=False,
                                 body=body, message=f"Nothing to commit on {branch}: the written files match HEAD")

    pushed = False
    url: str | None = None
    if push:
        gh.push_branch(repo, branch)
        pushed = True
    if create_pr:
        url = gh.create_pull_request(repo, title=pr_title, body=body, head=branch, base=base_branch, draft=draft)
        message = f"Opened {'draft ' if draft else ''}PR {url}"
    else:
        message = "Branch ready; PR creation skipped."
    return PullRequestResult(branch=branch, base=base_branch, url=url, committed=committed, pushed=pushed,
                             body=body, message=message)
