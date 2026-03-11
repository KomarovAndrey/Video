from __future__ import annotations

"""
Модельный скоринг учеников по рубрике на основе агрегированных признаков.

Ожидается, что для каждого ученика уже рассчитан `StudentFeatures`,
включающий как аудио-, так и видео-признаки.

Модели обучаются отдельно (см. модуль `training`) и сохраняются в:
    models/scorer_communication.joblib
    models/scorer_leadership.joblib
    models/scorer_self_regulation.joblib
    models/scorer_self_reflection.joblib
    models/scorer_critical_thinking.joblib

Если модель для критерия отсутствует, используется rule-based логика
из scoring.rule_based в качестве fallback.
"""

import os
from pathlib import Path
from typing import Dict, List

import joblib

from features import StudentFeatures
from rubric import Rubric
from .rule_based import (
    StudentScores,
    _score_communication,
    _score_critical_thinking,
    _score_leadership,
    _score_self_reflection,
    _score_self_regulation,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_MODELS_DIR = PROJECT_ROOT / "models"


def get_models_dir() -> Path:
    """Каталог моделей: SCORER_MODELS_DIR или models/ в корне проекта."""
    env = os.environ.get("SCORER_MODELS_DIR")
    if env:
        return Path(env)
    return _DEFAULT_MODELS_DIR


def _load_model(name: str):
    models_dir = get_models_dir()
    path = models_dir / f"scorer_{name}.joblib"
    if not path.exists():
        return None
    try:
        return joblib.load(path)
    except Exception:
        return None


def _load_feature_columns() -> List[str] | None:
    meta_path = get_models_dir() / "metadata.json"
    if not meta_path.exists():
        return None
    try:
        import json

        with open(meta_path, encoding="utf-8") as f:
            data = json.load(f)
        return data.get("feature_columns")
    except Exception:
        return None


MODELS_DIR = _DEFAULT_MODELS_DIR

_MODEL_COMMUNICATION = _load_model("communication")
_MODEL_LEADERSHIP = _load_model("leadership")
_MODEL_SELF_REG = _load_model("self_regulation")
_MODEL_SELF_REFL = _load_model("self_reflection")
_MODEL_CRIT = _load_model("critical_thinking")


def _features_to_vector(f: StudentFeatures) -> Dict[str, float]:
    """
    Преобразует StudentFeatures в словарь числовых признаков.
    Удобно подавать в tabular-модели (CatBoost/XGBoost/sklearn).
    """
    base = f.to_dict()
    return {str(k): float(v) for k, v in base.items()}


def _predict_level(model, f: StudentFeatures, fallback_fn) -> int:
    if model is None:
        return fallback_fn(f)
    try:
        x = _features_to_vector(f)
        if hasattr(model, "predict_dict"):
            pred = model.predict_dict(x)[0]
        else:
            feature_cols = _load_feature_columns()
            if feature_cols:
                vec = [[x.get(k, 0.0) for k in feature_cols]]
            else:
                keys = sorted(x.keys())
                vec = [[x[k] for k in keys]]
            pred = model.predict(vec)[0]
        level = int(round(float(pred)))
        return max(1, min(5, level))
    except Exception:
        return fallback_fn(f)


def score_students_model_based(
    rubric: Rubric,
    student_features: Dict[str, StudentFeatures],
) -> Dict[str, StudentScores]:
    """
    Возвращает уровни по всем критериям для каждого ученика,
    используя обученные модели, если они доступны.
    """
    scores: Dict[str, StudentScores] = {}
    for student_id, feats in student_features.items():
        levels: Dict[str, int] = {}
        levels["communication"] = _predict_level(
            _MODEL_COMMUNICATION, feats, _score_communication
        )
        levels["leadership"] = _predict_level(
            _MODEL_LEADERSHIP, feats, _score_leadership
        )
        levels["self_regulation"] = _predict_level(
            _MODEL_SELF_REG, feats, _score_self_regulation
        )
        levels["self_reflection"] = _predict_level(
            _MODEL_SELF_REFL, feats, _score_self_reflection
        )
        levels["critical_thinking"] = _predict_level(
            _MODEL_CRIT, feats, _score_critical_thinking
        )
        scores[student_id] = StudentScores(student_id=student_id, levels=levels)
    return scores


def get_criterion_feature_importances() -> Dict[str, Dict[str, float]]:
    """
    Возвращает важность признаков по каждому критерию для интерпретации оценок.
    Ключ — criterion_id, значение — словарь {имя_признака: важность}.
    Если модели или metadata отсутствуют, возвращается пустой словарь.
    """
    feature_cols = _load_feature_columns()
    if not feature_cols:
        return {}
    models_dir = get_models_dir()
    result: Dict[str, Dict[str, float]] = {}
    for crit in ("communication", "leadership", "self_regulation", "self_reflection", "critical_thinking"):
        path = models_dir / f"scorer_{crit}.joblib"
        if not path.exists():
            continue
        try:
            model = joblib.load(path)
            imp = getattr(model, "feature_importances_", None)
            if imp is not None and len(imp) == len(feature_cols):
                result[crit] = {name: float(imp[i]) for i, name in enumerate(feature_cols)}
            else:
                result[crit] = {}
        except Exception:
            continue
    return result


__all__ = ["score_students_model_based", "get_criterion_feature_importances", "get_models_dir"]

