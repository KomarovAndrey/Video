from __future__ import annotations

"""
Скрипт обучения модельного скоринга по рубрике.

Ожидается, что предварительно подготовлены:
- агрегированные признаки по ученикам (StudentFeatures.to_dict())
- рубричные оценки учителя по каждому критерию.

Формат входных данных (CSV):
    data/annotations/rubric_scores.csv
    колонки:
        video_id, student_id,
        leadership, communication, self_reflection, critical_thinking, self_regulation

    data/annotations/student_features.csv
    колонки:
        video_id, student_id, <все поля из StudentFeatures.to_dict()>

Результат:
    models/scorer_*.joblib  — по одной модели на критерий.
"""

from pathlib import Path
from typing import Dict, List

import joblib
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "annotations"
FEATURES_PATH = DATA_DIR / "student_features.csv"
SCORES_PATH = DATA_DIR / "rubric_scores.csv"
MODELS_DIR = PROJECT_ROOT / "models"


CRITERIA = [
    "communication",
    "leadership",
    "self_regulation",
    "self_reflection",
    "critical_thinking",
]


def load_features(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Не найден файл с признаками учеников: {path}. "
            f"Ожидается CSV с колонками: video_id, student_id, <признаки>."
        )
    df = pd.read_csv(path)
    required = {"video_id", "student_id"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"В {path} отсутствуют колонки: {', '.join(sorted(missing))}")
    return df


def load_scores(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Не найден файл с рубричными оценками: {path}. "
            f"Ожидается CSV с колонками: video_id, student_id, " + ", ".join(CRITERIA)
        )
    df = pd.read_csv(path)
    required = {"video_id", "student_id", *CRITERIA}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"В {path} отсутствуют колонки: {', '.join(sorted(missing))}")
    return df


def merge_features_scores(
    feats: pd.DataFrame, scores: pd.DataFrame
) -> pd.DataFrame:
    df = pd.merge(
        feats,
        scores,
        on=["video_id", "student_id"],
        how="inner",
        suffixes=("", "_target"),
    )
    if df.empty:
        raise ValueError("После объединения признаков и оценок не осталось ни одной строки.")
    return df


def train_models(df: pd.DataFrame) -> Dict[str, GradientBoostingRegressor]:
    feature_cols: List[str] = [
        c
        for c in df.columns
        if c
        not in {
            "video_id",
            "student_id",
            *CRITERIA,
        }
    ]
    X = df[feature_cols].values
    models: Dict[str, GradientBoostingRegressor] = {}

    for crit in CRITERIA:
        y = df[crit].astype(float).values
        model = GradientBoostingRegressor(random_state=42)
        model.fit(X, y)
        models[crit] = model

    return models


def save_models(models: Dict[str, GradientBoostingRegressor]) -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    for crit, model in models.items():
        path = MODELS_DIR / f"scorer_{crit}.joblib"
        joblib.dump(model, path)
        print(f"Модель для {crit} сохранена в {path}")


def main() -> None:
    feats = load_features(FEATURES_PATH)
    scores = load_scores(SCORES_PATH)
    df = merge_features_scores(feats, scores)
    models = train_models(df)
    save_models(models)


if __name__ == "__main__":
    main()

