from __future__ import annotations

from pathlib import Path
from typing import Optional

import joblib

from features import ActivityType


_MODEL_PATH = Path("models") / "activity_classifier.joblib"


def _rule_based_classify(text: str) -> ActivityType:
    """
    Простейший эвристический классификатор активности по тексту.
    Используется как fallback, если обученная модель недоступна.
    """
    t = text.lower()
    if "?" in t or any(k in t for k in ["почему", "зачем", "как", "можно ли"]):
        return "asking_question"
    if any(k in t for k in ["я думаю", "предлагаю", "можно сделать", "идея"]):
        return "giving_idea"
    if any(k in t for k in ["не соглас", "я считаю, что это не так"]):
        return "disagreeing"
    if any(k in t for k in ["соглас", "правильно", "я тоже так думаю"]):
        return "supporting_peer"
    if any(k in t for k in ["не буду", "не хочу", "отстань"]):
        return "off_task"
    return "answering"


def _load_model(model_path: Path = _MODEL_PATH):
    if not model_path.exists():
        return None
    try:
        return joblib.load(model_path)
    except Exception:
        return None


_MODEL = _load_model()


def classify_activity(text: str) -> ActivityType:
    """
    Классификация типа активности по тексту высказывания.

    Если обученная модель доступна (models/activity_classifier.joblib),
    используется она, иначе — резервный rule-based классификатор.
    """
    if not text:
        return "silent"
    if _MODEL is None:
        return _rule_based_classify(text)

    try:
        pred = _MODEL.predict([text])[0]
        # Защитимся от неожиданных значений.
        if pred in (
            "silent",
            "listening",
            "asking_question",
            "answering",
            "giving_idea",
            "disagreeing",
            "supporting_peer",
            "off_task",
        ):
            return pred  # type: ignore[return-value]
    except Exception:
        pass
    return _rule_based_classify(text)


__all__ = ["classify_activity"]

