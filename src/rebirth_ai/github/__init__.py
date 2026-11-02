"""GitHub branch / PR automation (Phase 5)."""

from rebirth_ai.github.client import GitHubError, auth_status, require_auth
from rebirth_ai.github.pr import PullRequestResult, open_migration_pr

__all__ = [
    "GitHubError",
    "PullRequestResult",
    "auth_status",
    "open_migration_pr",
    "require_auth",
]
