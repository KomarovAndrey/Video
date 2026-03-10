from __future__ import annotations

"""
Скрипт обучения текстового классификатора типов активности ученика.

Ожидает входные данные в формате CSV:
- data/annotations/utterances.csv
  - utterance_id, student_id, start, end, text, activity_type

Результат обучения сохраняется в:
- models/activity_classifier.joblib

Запуск (из корня проекта):
    python -m training.train_activity_classifier
"""

from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "annotations" / "utterances.csv"
MODEL_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODEL_DIR / "activity_classifier.joblib"


def load_dataset(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Не найден файл с разметкой реплик: {path}. "
            f"Ожидается CSV с колонками: utterance_id, student_id, start, end, text, activity_type."
        )
    df = pd.read_csv(path)
    required = {"text", "activity_type"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"В файле {path} отсутствуют колонки: {', '.join(sorted(missing))}")
    df = df.dropna(subset=["text", "activity_type"])
    df["text"] = df["text"].astype(str)
    df["activity_type"] = df["activity_type"].astype(str)
    return df


def build_pipeline() -> Pipeline:
    """
    Простой baseline: TF-IDF по словам + логистическая регрессия.
    """
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    ngram_range=(1, 2),
                    max_features=50000,
                    min_df=2,
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    max_iter=200,
                    n_jobs=-1,
                    verbose=0,
                    multi_class="auto",
                ),
            ),
        ]
    )


def train_and_save() -> None:
    df = load_dataset(DATA_PATH)
    X = df["text"].tolist()
    y = df["activity_type"].tolist()

    pipe = build_pipeline()
    pipe.fit(X, y)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, MODEL_PATH)
    print(f"Модель сохранена в {MODEL_PATH}")


if __name__ == "__main__":
    train_and_save()

