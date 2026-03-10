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

from pathlib import Path
from typing import Dict

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
MODELS_DIR = PROJECT_ROOT / "models"


def _load_model(name: str):
    path = MODELS_DIR / f"scorer_{name}.joblib"
    if not path.exists():
        return None
    try:
        return joblib.load(path)
    except Exception:
        return None


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
        # Ожидаем, что модель умеет работать с dict или с упорядоченным списком признаков.
        # Базовая реализация рассчитывает по отсортированным ключам.
        if hasattr(model, "predict_dict"):
            pred = model.predict_dict(x)[0]
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


__all__ = ["score_students_model_based"]

