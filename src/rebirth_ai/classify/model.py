"""TF-IDF + RandomForest error classifier."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from rebirth_ai.classify.features import ERROR_CLASSES, normalize_log

DEFAULT_MODEL_DIR = Path("data/models/error_classifier")


@dataclass(frozen=True)
class TrainReport:
    train_size: int
    test_size: int
    accuracy: float
    report: str
    model_dir: str


class ErrorClassifier:
    def __init__(self, pipeline: Pipeline | None = None) -> None:
        self.pipeline = pipeline

    @staticmethod
    def build_pipeline() -> Pipeline:
        return Pipeline(
            steps=[
                (
                    "tfidf",
                    TfidfVectorizer(
                        lowercase=True,
                        ngram_range=(1, 2),
                        min_df=1,
                        max_features=5000,
                    ),
                ),
                (
                    "clf",
                    LogisticRegression(
                        max_iter=2000,
                        random_state=42,
                        class_weight="balanced",
                        C=2.0,
                    ),
                ),
            ]
        )

    @classmethod
    def train_from_jsonl(
        cls,
        dataset_path: Path,
        *,
        model_dir: Path = DEFAULT_MODEL_DIR,
        test_size: float = 0.25,
    ) -> tuple[ErrorClassifier, TrainReport]:
        texts: list[str] = []
        labels: list[str] = []
        with dataset_path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                label = str(row["label"])
                if label not in ERROR_CLASSES:
                    raise ValueError(f"Unknown label {label!r}")
                texts.append(normalize_log(str(row["text"])))
                labels.append(label)

        if len(texts) < 8:
            raise ValueError("Need at least 8 labeled examples to train")

        x_train, x_test, y_train, y_test = train_test_split(
            texts,
            labels,
            test_size=test_size,
            random_state=42,
            stratify=labels,
        )
        pipe = cls.build_pipeline()
        pipe.fit(x_train, y_train)
        preds = pipe.predict(x_test)
        acc = float(accuracy_score(y_test, preds))
        report = classification_report(y_test, preds, digits=3)

        model_dir = Path(model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(pipe, model_dir / "pipeline.joblib")
        (model_dir / "labels.json").write_text(
            json.dumps(list(ERROR_CLASSES), indent=2),
            encoding="utf-8",
        )
        (model_dir / "metrics.json").write_text(
            json.dumps({"accuracy": acc, "train_size": len(x_train), "test_size": len(x_test)}, indent=2),
            encoding="utf-8",
        )

        clf = cls(pipe)
        return clf, TrainReport(
            train_size=len(x_train),
            test_size=len(x_test),
            accuracy=acc,
            report=report,
            model_dir=str(model_dir),
        )

    @classmethod
    def load(cls, model_dir: Path = DEFAULT_MODEL_DIR) -> ErrorClassifier:
        path = Path(model_dir) / "pipeline.joblib"
        if not path.exists():
            raise FileNotFoundError(
                f"Classifier not found at {path}. Train with: rebirth train-classifier"
            )
        return cls(joblib.load(path))

    def predict(self, log_text: str) -> str:
        if self.pipeline is None:
            raise RuntimeError("Classifier pipeline is not loaded")
        pred = self.pipeline.predict([normalize_log(log_text)])[0]
        return str(pred)

    def predict_proba(self, log_text: str) -> dict[str, float]:
        if self.pipeline is None:
            raise RuntimeError("Classifier pipeline is not loaded")
        if not hasattr(self.pipeline, "predict_proba"):
            return {self.predict(log_text): 1.0}
        probs = self.pipeline.predict_proba([normalize_log(log_text)])[0]
        classes = list(self.pipeline.classes_)
        return {str(c): float(p) for c, p in zip(classes, probs, strict=True)}
