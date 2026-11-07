"""Create a branch, commit migration successes, and open a GitHub PR."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from rebirth_ai.github import client as gh
from rebirth_ai.github.client import GitHubError
from rebirth_ai.graph.batch import BatchReport, render_pr_body
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
    repo: Path,
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
    """
    Stage successful writes (+ optional report JSON), commit, push, open PR.

    Failed files are listed in the PR body; they are not required to block the PR.
    ``base=None`` auto-detects the repo default branch (main/master).
    """
    repo = gh.git_root(repo)
    base_branch = base or gh.detect_default_branch(repo)
    body = render_pr_body(report)
    pr_title = title or f"Rebirth AI: migrate {len(report.succeeded)} file(s)"

    if paths is None:
        paths = [r.relative_file for r in report.succeeded if r.wrote]
    if report_path is not None:
        # Prefer path relative to repo when possible
        try:
            rel_report = str(report_path.resolve().relative_to(repo))
        except ValueError:
            rel_report = str(report_path)
        if rel_report not in paths:
            paths = [*paths, rel_report]

    paths, blocked = filter_secret_paths(paths)
    if blocked:
        raise GitHubError(
            "Refusing to stage secret/credential paths in migration PR: "
            + ", ".join(blocked)
        )

    gh.create_branch(repo, branch)
    if paths:
        gh.stage_paths(repo, paths)
    committed = gh.commit(
        repo,
        f"rebirth: migrate {len(report.succeeded)} file(s)\n\n"
        f"Failed/review: {len(report.failed)}",
    )

    pushed = False
    url: str | None = None
    message = ""

    if not committed and not paths:
        message = "Nothing to commit (no written successes)."
        return PullRequestResult(
            branch=branch,
            base=base_branch,
            url=None,
            committed=False,
            pushed=False,
            body=body,
            message=message,
        )

    if push:
        gh.push_branch(repo, branch)
        pushed = True

    if create_pr:
        url = gh.create_pull_request(
            repo,
            title=pr_title,
            body=body,
            head=branch,
            base=base_branch,
            draft=draft,
        )
        message = f"Opened {'draft ' if draft else ''}PR {url}"
    else:
        message = "Branch ready; PR creation skipped."

    return PullRequestResult(
        branch=branch,
        base=base_branch,
        url=url,
        committed=committed,
        pushed=pushed,
        body=body,
        message=message,
    )
