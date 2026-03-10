from __future__ import annotations

from collections import defaultdict
from typing import Dict, Iterable, List, Tuple

from .models import ActivityType, StudentFeatures, Utterance, VideoActivitySegment


def aggregate_student_features(
    utterances: Iterable[Utterance],
    video_segments: Iterable[VideoActivitySegment],
) -> Dict[str, StudentFeatures]:
    """
    Агрегирует признаки на уровне ученика на основе реплик и видеосегментов.
    """
    features: Dict[str, StudentFeatures] = {}

    def get_student(student_id: str) -> StudentFeatures:
        if student_id not in features:
            features[student_id] = StudentFeatures(student_id=student_id)
        return features[student_id]

    for utt in utterances:
        s = get_student(utt.student_id)
        duration = max(0.0, float(utt.end - utt.start))
        s.total_speaking_time += duration
        s.num_utterances += 1
        s.activity_breakdown[utt.activity_type] += 1

        if utt.activity_type == "asking_question":
            s.num_questions += 1
        elif utt.activity_type == "giving_idea":
            s.num_ideas += 1
        elif utt.activity_type == "disagreeing":
            s.num_disagreements += 1
        elif utt.activity_type == "supporting_peer":
            s.num_supports += 1
        elif utt.activity_type == "off_task":
            s.num_off_task_events += 1

    # Видео-сегменты: считаем количество событий и грубые оценки внимания.
    # Здесь пока нет позы/мимики, но структура позволяет добавить их позже.
    video_counts: Dict[str, Dict[str, int]] = defaultdict(
        lambda: defaultdict(int)
    )  # student_id -> event_type -> count

    for seg in video_segments:
        s = get_student(seg.student_id)
        if seg.is_hand_raised:
            s.hand_raise_count += 1
            video_counts[seg.student_id]["hand_raise"] += 1
        if seg.is_addressing_teacher:
            s.address_teacher_count += 1
            video_counts[seg.student_id]["address_teacher"] += 1
        if seg.is_addressing_class:
            s.address_class_count += 1
            video_counts[seg.student_id]["address_class"] += 1
        if seg.is_off_task:
            s.num_off_task_events += 1
            s.video_off_task_events += 1
            video_counts[seg.student_id]["video_off_task"] += 1

    # Нормируем простые видео-оценки внимания и жестовой активности.
    for student_id, events in video_counts.items():
        s = get_student(student_id)
        total_events = sum(events.values()) or 1
        attentive_events = events.get("address_teacher", 0) + events.get(
            "address_class", 0
        )
        s.video_attention_score = attentive_events / total_events
        s.gesture_activity_score = events.get("hand_raise", 0) / total_events

    return features

