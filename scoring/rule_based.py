from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

from features import StudentFeatures
from rubric import Rubric


@dataclass
class StudentScores:
    student_id: str
    levels: Dict[str, int]  # criterion_id -> level


def _score_communication(f: StudentFeatures) -> int:
    if f.num_utterances == 0:
        return 1
    if f.total_speaking_time < 10 and f.num_utterances <= 2:
        return 2
    if f.num_questions + f.num_ideas >= 3 and f.address_class_count + f.address_teacher_count >= 3:
        return 4
    if f.num_questions + f.num_ideas >= 6 and f.address_class_count >= 3:
        return 5
    return 3


def _score_leadership(f: StudentFeatures) -> int:
    initiatives = f.num_ideas + f.num_questions + f.address_class_count
    if initiatives == 0:
        return 1
    if initiatives <= 2:
        return 2
    if initiatives <= 5:
        return 3
    if initiatives <= 8:
        return 4
    return 5


def _score_self_regulation(f: StudentFeatures) -> int:
    if f.num_off_task_events >= 8:
        return 1
    if f.num_off_task_events >= 4:
        return 2
    if f.num_off_task_events >= 2:
        return 3
    if f.num_off_task_events == 1:
        return 4
    return 5


def _score_self_reflection(f: StudentFeatures) -> int:
    # Пока нет прямых индикаторов рефлексии, используем прокси:
    # вопросы + идеи как сигнал осмысленной работы.
    refl = f.num_questions + f.num_ideas
    if refl == 0:
        return 1
    if refl <= 2:
        return 2
    if refl <= 4:
        return 3
    if refl <= 6:
        return 4
    return 5


def _score_critical_thinking(f: StudentFeatures) -> int:
    # Используем несогласия и вопросы как сигнал критического мышления.
    crit = f.num_disagreements + f.num_questions
    if crit == 0:
        return 1
    if crit == 1:
        return 2
    if crit <= 3:
        return 3
    if crit <= 6:
        return 4
    return 5


def score_students(
    rubric: Rubric,
    student_features: Dict[str, StudentFeatures],
) -> Dict[str, StudentScores]:
    """
    Возвращает уровни по всем критериям для каждого ученика.
    """
    scores: Dict[str, StudentScores] = {}
    for student_id, feats in student_features.items():
        levels: Dict[str, int] = {}
        levels["communication"] = _score_communication(feats)
        levels["leadership"] = _score_leadership(feats)
        levels["self_regulation"] = _score_self_regulation(feats)
        levels["self_reflection"] = _score_self_reflection(feats)
        levels["critical_thinking"] = _score_critical_thinking(feats)
        scores[student_id] = StudentScores(student_id=student_id, levels=levels)
    return scores

