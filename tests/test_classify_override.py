"""Heuristic DependencyMissing override (Phase 7 express path)."""

from rebirth_ai.graph.nodes import _heuristic_error_class, classify_error


def test_heuristic_detects_esm_missing_package() -> None:
    logs = (
        "Error [ERR_MODULE_NOT_FOUND]: Cannot find package 'ms' "
        "imported from /workspace/src/duration.js"
    )
    assert _heuristic_error_class(logs) == "DependencyMissing"


def test_classify_error_overrides_to_dependency_missing() -> None:
    state = classify_error(
        {
            "test_logs": (
                "Error [ERR_MODULE_NOT_FOUND]: Cannot find package 'ms' "
                "imported from /workspace/src/duration.js\n"
            ),
            "trace": [],
        }
    )
    assert state["error_class"] == "DependencyMissing"
    assert "DependencyMissing" in state["trace"][-1]


import pytest  # noqa: E402


@pytest.mark.parametrize("log,want", [
    ("./homedir.go:165:1: syntax error: unexpected EOF, expected }", "SyntaxError"),
    ("error: expected one of `;` or `}`, found `let`\n --> src/lib.rs:3:5", "SyntaxError"),
    ("Calc.java:7: error: ';' expected", "SyntaxError"),
    ("Calculator.cs(12,5): error CS1002: ; expected", "SyntaxError"),
    ("math_ops.cpp:4:3: error: expected ';' before 'return'", "SyntaxError"),
    ("error[E0308]: mismatched types\n expected `i32`, found `&str`", "TypeMismatch"),
    ("./a.go:9:13: cannot use s (variable of type string) as int value in return statement", "TypeMismatch"),
    ("Calc.java:9: error: incompatible types: String cannot be converted to int", "TypeMismatch"),
    ("FAIL: TestAdd (0.00s)\n    a_test.go:12: got 3, want 4", None),
    ("AssertionError: expected 4 to equal 5\n    at Context.<anonymous> (test/a.test.js:5:12)", None),
    ("  1) reports a syntax error in the config\n     Error: expected false to be true", None),
    ("Sources/MathOps/MathOps.swift:3:5: error: expected '}' in class", "SyntaxError"),
])
def test_compiled_language_overrides(log, want) -> None:
    from rebirth_ai.graph.nodes import _heuristic_error_class

    assert _heuristic_error_class(log) == want


def test_low_confidence_dependency_guess_does_not_route_to_install(monkeypatch) -> None:
    from rebirth_ai.graph import nodes

    class Clf:
        def predict_proba(self, logs):
            return {"DependencyMissing": 0.34, "TestLogicFailure": 0.33, "SyntaxError": 0.2, "TypeMismatch": 0.13}

    monkeypatch.setattr(nodes.ErrorClassifier, "load", classmethod(lambda cls, *a, **k: Clf()))
    out = nodes.classify_error({"test_logs": "FAIL: TestAdd\n a_test.go:12: got 3, want 4", "trace": []})
    assert out["error_class"] == "TestLogicFailure" and out["error_unknown"]
    out = nodes.classify_error({"test_logs": "Error: Cannot find package 'ms' imported", "trace": []})
    assert out["error_class"] == "DependencyMissing"
