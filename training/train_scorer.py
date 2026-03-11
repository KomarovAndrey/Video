from __future__ import annotations

"""
Скрипт обучения модельного скоринга по рубрике.

Ожидается, что предварительно подготовлены:
- агрегированные признаки по ученикам (StudentFeatures.to_dict())
- рубричные оценки учителя по каждому критерию.

Формат входных данных (CSV):
    data/annotations/rubric_scores.csv
    data/annotations/student_features.csv

Результат:
    models/scorer_*.joblib  — по одной модели на критерий.
    models/metadata.json   — список признаков (для интерпретации и калибровки).

Калибровка под школу: укажите --models-dir и пути к своим CSV.
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_val_predict, KFold

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "annotations"
FEATURES_PATH = DATA_DIR / "student_features.csv"
SCORES_PATH = DATA_DIR / "rubric_scores.csv"
MODELS_DIR = PROJECT_ROOT / "models"

try:
    from catboost import CatBoostRegressor

    _HAS_CATBOOST = True
except Exception:
    _HAS_CATBOOST = False

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


def _get_feature_columns(df: pd.DataFrame) -> List[str]:
    return [
        c
        for c in df.columns
        if c not in {"video_id", "student_id", *CRITERIA}
    ]


def train_models(
    df: pd.DataFrame,
    feature_cols: List[str],
    model_type: str = "gb",
) -> Dict[str, Any]:
    X = df[feature_cols].values
    models: Dict[str, Any] = {}

    for crit in CRITERIA:
        y = df[crit].astype(float).values
        if model_type == "catboost" and _HAS_CATBOOST:
            model = CatBoostRegressor(random_state=42, verbose=0)
        else:
            model = GradientBoostingRegressor(random_state=42)
        model.fit(X, y)
        models[crit] = model

    return models


def report_mae(df: pd.DataFrame, feature_cols: List[str], models: Dict[str, Any]) -> None:
    """Кросс-валидация и вывод MAE по каждому критерию."""
    X = df[feature_cols].values
    kf = KFold(n_splits=min(5, len(df) // 2 or 1), shuffle=True, random_state=42)
    for crit in CRITERIA:
        y = df[crit].astype(float).values
        pred = cross_val_predict(models[crit], X, y, cv=kf)
        mae = float(np.mean(np.abs(np.array(pred) - y)))
        print(f"  {crit}: MAE = {mae:.3f}")


def save_models(
    models: Dict[str, Any],
    models_dir: Path,
    feature_cols: List[str],
) -> None:
    models_dir.mkdir(parents=True, exist_ok=True)
    for crit, model in models.items():
        path = models_dir / f"scorer_{crit}.joblib"
        joblib.dump(model, path)
        print(f"Модель для {crit} сохранена в {path}")
    meta_path = models_dir / "metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump({"feature_columns": feature_cols}, f, ensure_ascii=False)
    print(f"Метаданные сохранены в {meta_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Обучение скореров по рубрике")
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=MODELS_DIR,
        help="Каталог для сохранения моделей (калибровка под школу)",
    )
    parser.add_argument(
        "--features",
        type=Path,
        default=FEATURES_PATH,
        help="Путь к CSV с признаками учеников",
    )
    parser.add_argument(
        "--scores",
        type=Path,
        default=SCORES_PATH,
        help="Путь к CSV с рубричными оценками",
    )
    parser.add_argument(
        "--model-type",
        choices=["gb", "catboost"],
        default="gb",
        help="Тип модели (catboost при установленном пакете)",
    )
    parser.add_argument(
        "--no-cv",
        action="store_true",
        help="Не выводить MAE по кросс-валидации",
    )
    args = parser.parse_args()

    feats = load_features(args.features)
    scores = load_scores(args.scores)
    df = merge_features_scores(feats, scores)
    feature_cols = _get_feature_columns(df)
    models = train_models(df, feature_cols, model_type=args.model_type)
    if not args.no_cv:
        print("MAE (кросс-валидация):")
        report_mae(df, feature_cols, models)
    save_models(models, args.models_dir, feature_cols)


if __name__ == "__main__":
    main()

