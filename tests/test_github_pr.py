"""`migrate-repo --open-pr` (Phase 19 bug hunt): a real git repo + local bare remote, a fake `gh`."""

from __future__ import annotations

import os
import stat
import subprocess
from pathlib import Path

import pytest

from rebirth_ai.github import GitHubError, open_migration_pr
from rebirth_ai.graph.batch import BatchReport, FileMigrationResult


def git(cwd: Path, *args: str) -> str:
    return subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True).stdout


@pytest.fixture
def repo(tmp_path, monkeypatch) -> Path:
    remote = tmp_path / "origin.git"
    git(tmp_path, "init", "-q", "--bare", "-b", "main", str(remote))
    work = tmp_path / "work"
    git(tmp_path, "init", "-q", "-b", "main", str(work))
    for k, v in (("user.email", "t@example.com"), ("user.name", "t"), ("commit.gpgsign", "false")):
        git(work, "config", k, v)
    (work / "pkg" / "src").mkdir(parents=True)
    (work / "pkg" / "src" / "a.js").write_text("var a = 1;\n")
    (work / "notes.txt").write_text("mine\n")
    git(work, "add", ".")
    git(work, "commit", "-q", "-m", "init")
    git(work, "remote", "add", "origin", "https://github.com/me/throwaway.git")
    git(work, "config", "remote.origin.pushurl", str(remote))  # pushes go to the local bare repo
    git(work, "push", "-q", "origin", "main")
    bindir = tmp_path / "bin"
    bindir.mkdir()
    gh = bindir / "gh"
    gh.write_text(f"""#!/usr/bin/env bash
echo "$@" >> "{tmp_path}/gh.log"
case "$1 $2" in
  "auth status") echo "github.com: Logged in to github.com account me (keyring)" ;;
  "pr create") cat "$(echo "$@" | sed -E 's/.*--body-file ([^ ]+).*/\\1/')" > "{tmp_path}/body.md"
               echo "https://github.com/me/throwaway/pull/1" ;;
  *) exit 1 ;;
esac
""")
    gh.chmod(gh.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setenv("PATH", f"{bindir}{os.pathsep}{os.environ['PATH']}")
    return work


def report(project: Path, wrote: bool = True) -> BatchReport:
    return BatchReport(project_dir=str(project), objective="modernize", results=[
        FileMigrationResult(relative_file="src/a.js", relative_test="package.json", status="success", wrote=wrote),
        FileMigrationResult(relative_file="src/b.js", relative_test=None, status="human_review",
                            error_class="TestLogicFailure", test_logs_excerpt="token ghp_abcdefghijklmnopqrstuvwxyz0123456789"),
    ])


def test_pr_commits_only_the_written_file_of_the_project(repo) -> None:
    project = repo / "pkg"
    (project / "src" / "a.js").write_text("const a = 1;\n")  # what --write put there
    (repo / "notes.txt").write_text("staged by me\n")
    git(repo, "add", "notes.txt")  # the user's own staged change must stay out of the commit
    res = open_migration_pr(project, report(project), branch="rebirth/migrate")
    assert res.url == "https://github.com/me/throwaway/pull/1" and res.committed and res.pushed
    assert git(repo, "show", "--name-only", "--format=", "HEAD").split() == ["pkg/src/a.js"]
    assert "notes.txt" in git(repo, "diff", "--cached", "--name-only")  # still staged, not committed
    assert "rebirth/migrate" in git(repo, "ls-remote", "--heads", str(repo.parent / "origin.git"))
    body = (repo.parent / "body.md").read_text()
    assert "`work/pkg`" in body and str(repo.parent) not in body  # no local paths in the PR
    assert "ghp_abcdefghijklmnopqrstuvwxyz0123456789" not in body  # redacted
    assert "evidence, not proof" in body
    with pytest.raises(GitHubError, match="already exists"):  # -B used to reset it silently
        open_migration_pr(project, report(project), branch="rebirth/migrate")


def test_nothing_written_creates_no_branch(repo) -> None:
    res = open_migration_pr(repo / "pkg", report(repo / "pkg", wrote=False))
    assert not res.committed and "Nothing to commit" in res.message
    assert "rebirth/migrate" not in git(repo, "branch")
    assert not (repo.parent / "gh.log").exists() or "pr create" not in (repo.parent / "gh.log").read_text()


def test_secret_paths_are_refused(repo) -> None:
    with pytest.raises(GitHubError, match="secret"):
        open_migration_pr(repo, report(repo), paths=[".env"])
