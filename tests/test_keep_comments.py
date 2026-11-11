"""Phase 19: a passing draft that deletes the original's comments goes back to the model."""

from __future__ import annotations

import pytest

from rebirth_ai.graph import nodes

GO = """package h

// Dir returns the home directory.
func Dir() string {
	// First prefer the HOME environment variable
	if h := env("HOME"); h != "" {
		return h
	}
	// If that fails, try the shell
	// (plan9 uses lowercase names)
	return shell()
}
"""
GO_NO_COMMENTS = """package h

func Dir() string {
	if h := env("HOME"); h != "" {
		return h
	}
	return shell()
}
"""


def test_comment_loss_threshold(monkeypatch) -> None:
    assert nodes.comments_lost(GO, GO_NO_COMMENTS, "go") == (4, 4)
    one_less = GO.replace("\t// (plan9 uses lowercase names)\n", "")
    assert nodes.comments_lost(GO, one_less, "go") == (0, 4)  # 1 line: allowed (it may be obsolete)
    py = '"""Doc."""\n# a\n# b\n# c\ndef f():\n    """Doc."""\n    return 1\n'
    assert nodes.comments_lost(py, "def f():\n    return 1\n", "python")[0] == 7
    monkeypatch.setenv("REBIRTH_KEEP_COMMENTS", "0")
    assert nodes.comments_lost(GO, GO_NO_COMMENTS, "go") == (0, 0)


class Res:
    ok, exit_code, duration_sec, stdout, stderr = True, 0, 1.0, "ok", ""


@pytest.fixture
def passing_sandbox(monkeypatch):
    class Runner:
        def __init__(self, adapter):
            pass

        def setup_command(self, *a):
            return None

        def run_project(self, *a, **k):
            return Res()

    monkeypatch.setattr(nodes, "SandboxRunner", Runner)


def state(draft, iteration=1):
    return {"project_dir": ".", "file_path": "h.go", "relative_file": "h.go", "relative_test": "",
            "original_code": GO, "current_draft": draft, "adapter_id": "go", "iteration_count": iteration, "trace": []}


def test_passing_draft_without_comments_is_sent_back(passing_sandbox) -> None:
    out = nodes.execute_tests(state(GO_NO_COMMENTS))
    assert out["status"] == "running" and out["test_logs"].startswith(nodes.KEPT_COMMENTS_NOTE)
    assert out["trace"][-1].endswith("but 4 of 4 comment lines deleted")
    cls = nodes.classify_error({**out, "trace": []})
    assert cls["error_class"] == "TestLogicFailure"  # regenerate, never install_dependency
    assert nodes.execute_tests(state(GO))["status"] == "success"
    assert nodes.execute_tests(state(GO_NO_COMMENTS, iteration=0))["status"] == "success"  # a seed draft
