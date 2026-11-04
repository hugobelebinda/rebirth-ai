"""Thin GitHub / git helpers via the ``gh`` CLI (Phase 5)."""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path


class GitHubError(RuntimeError):
    """Raised when ``gh`` / git automation fails."""


@dataclass(frozen=True)
class GhAuthStatus:
    ok: bool
    login: str | None
    detail: str


def _run(
    args: list[str],
    *,
    cwd: Path | None = None,
    check: bool = True,
    input_text: str | None = None,
) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            args,
            cwd=str(cwd) if cwd else None,
            check=check,
            text=True,
            input=input_text,
            capture_output=True,
        )
    except FileNotFoundError as exc:
        raise GitHubError(f"Command not found: {args[0]}") from exc
    except subprocess.CalledProcessError as exc:
        err = (exc.stderr or exc.stdout or "").strip()
        raise GitHubError(f"{' '.join(args)} failed: {err}") from exc


def ensure_gh() -> Path:
    path = shutil.which("gh")
    if not path:
        raise GitHubError("`gh` CLI not found on PATH. Install GitHub CLI.")
    return Path(path)


def auth_status() -> GhAuthStatus:
    ensure_gh()
    proc = _run(["gh", "auth", "status"], check=False)
    text = (proc.stdout or "") + (proc.stderr or "")
    if proc.returncode != 0:
        return GhAuthStatus(ok=False, login=None, detail=text.strip())
    login = None
    for line in text.splitlines():
        if "Logged in to github.com account" in line or "account" in line.lower():
            parts = line.split()
            for i, p in enumerate(parts):
                if p == "account" and i + 1 < len(parts):
                    login = parts[i + 1].strip("()")
                    break
    return GhAuthStatus(ok=True, login=login, detail=text.strip())


def require_auth() -> GhAuthStatus:
    status = auth_status()
    if not status.ok:
        raise GitHubError(
            "GitHub CLI is not authenticated. Run: gh auth login -h github.com\n"
            f"{status.detail}"
        )
    return status


def git_root(path: Path) -> Path:
    proc = _run(["git", "rev-parse", "--show-toplevel"], cwd=path)
    return Path(proc.stdout.strip())


def current_branch(repo: Path) -> str:
    proc = _run(["git", "branch", "--show-current"], cwd=repo)
    return proc.stdout.strip()


def detect_default_branch(repo: Path, *, remote: str = "origin") -> str:
    """Detect repo default branch (``main`` / ``master`` / remote HEAD)."""
    # origin/HEAD → origin/main
    sym = _run(
        ["git", "symbolic-ref", f"refs/remotes/{remote}/HEAD"],
        cwd=repo,
        check=False,
    )
    if sym.returncode == 0:
        ref = sym.stdout.strip()  # refs/remotes/origin/main
        prefix = f"refs/remotes/{remote}/"
        if ref.startswith(prefix):
            return ref[len(prefix) :]

    # gh metadata when authenticated
    try:
        ensure_gh()
        proc = _run(
            ["gh", "repo", "view", "--json", "defaultBranchRef", "--jq", ".defaultBranchRef.name"],
            cwd=repo,
            check=False,
        )
        name = (proc.stdout or "").strip()
        if proc.returncode == 0 and name:
            return name
    except GitHubError:
        pass

    for candidate in ("main", "master"):
        probe = _run(
            ["git", "rev-parse", "--verify", f"{remote}/{candidate}"],
            cwd=repo,
            check=False,
        )
        if probe.returncode == 0:
            return candidate
    return "main"


def create_branch(repo: Path, name: str, *, from_ref: str | None = None) -> None:
    args = ["git", "checkout", "-B", name]
    if from_ref:
        args.append(from_ref)
    _run(args, cwd=repo)


def stage_paths(repo: Path, paths: list[str]) -> None:
    if not paths:
        return
    _run(["git", "add", "--", *paths], cwd=repo)


def commit(repo: Path, message: str) -> bool:
    """Create a commit. Returns False if there was nothing to commit."""
    staged = _run(["git", "diff", "--cached", "--quiet"], cwd=repo, check=False)
    if staged.returncode == 0:
        return False
    _run(["git", "commit", "-m", message], cwd=repo)
    return True


def push_branch(repo: Path, branch: str, *, remote: str = "origin") -> None:
    _run(["git", "push", "-u", remote, branch], cwd=repo)


def repo_slug_from_remote(repo: Path, *, remote: str = "origin") -> str:
    proc = _run(["git", "remote", "get-url", remote], cwd=repo)
    url = proc.stdout.strip()
    # https://github.com/owner/repo.git or git@github.com:owner/repo.git
    if url.endswith(".git"):
        url = url[:-4]
    if "github.com" in url:
        if url.startswith("git@"):
            return url.split(":", 1)[1]
        return url.split("github.com/", 1)[1].strip("/")
    raise GitHubError(f"Cannot parse GitHub slug from remote URL: {url}")


def create_pull_request(
    repo: Path,
    *,
    title: str,
    body: str,
    head: str,
    base: str = "main",
    draft: bool = False,
) -> str:
    """Open a PR with ``gh``; returns the PR URL."""
    require_auth()
    # Write body to a temp file — avoids shell/flag issues with long markdown
    import tempfile

    with tempfile.NamedTemporaryFile(
        "w", suffix=".md", delete=False, encoding="utf-8"
    ) as fh:
        fh.write(body)
        body_file = fh.name
    try:
        args = [
            "gh",
            "pr",
            "create",
            "--title",
            title,
            "--body-file",
            body_file,
            "--base",
            base,
            "--head",
            head,
        ]
        if draft:
            args.append("--draft")
        proc = _run(args, cwd=repo)
    finally:
        Path(body_file).unlink(missing_ok=True)

    text = (proc.stdout or "").strip() or (proc.stderr or "").strip()
    for token in text.split():
        if token.startswith("https://github.com/") and "/pull/" in token:
            return token
    if text.startswith("http"):
        return text.split()[0]
    raise GitHubError(f"Could not parse PR URL from gh output: {text!r}")
