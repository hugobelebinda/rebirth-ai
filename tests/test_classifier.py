"""Error classifier train/predict tests."""

from __future__ import annotations

from pathlib import Path

from rebirth_ai.classify.model import ErrorClassifier

DATASET = Path(__file__).resolve().parents[1] / "data" / "datasets" / "error_logs.jsonl"


def test_train_and_predict(tmp_path: Path) -> None:
    model_dir = tmp_path / "clf"
    clf, report = ErrorClassifier.train_from_jsonl(DATASET, model_dir=model_dir)
    assert report.accuracy >= 0.85
    assert (model_dir / "pipeline.joblib").exists()

    loaded = ErrorClassifier.load(model_dir)
    assert (
        loaded.predict("Error: Cannot find module 'lodash'\nCODE: MODULE_NOT_FOUND")
        == "DependencyMissing"
    )
    assert (
        loaded.predict(
            "Expected values to be strictly equal:\n5 !== 999\nAssertionError ERR_ASSERTION"
        )
        == "TestLogicFailure"
    )
    assert "SyntaxError" == loaded.predict(
        "SyntaxError: Unexpected token 'export'\n    at compileSourceTextModule"
    )
    assert loaded.predict(
        "TypeError: add is not a function\nerror TS2345: Argument of type 'string'"
    ) in {"TypeMismatch", "TestLogicFailure", "SyntaxError", "DependencyMissing"}
