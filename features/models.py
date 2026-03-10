from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Literal, Tuple


ActivityType = Literal[
    "silent",
    "listening",
    "asking_question",
    "answering",
    "giving_idea",
    "disagreeing",
    "supporting_peer",
    "off_task",
]


@dataclass
class Utterance:
    student_id: str
    start: float
    end: float
    text: str
    activity_type: ActivityType


@dataclass
class VideoActivitySegment:
    student_id: str
    start: float
    end: float
    is_hand_raised: bool = False
    is_addressing_teacher: bool = False
    is_addressing_class: bool = False
    is_off_task: bool = False


@dataclass
class StudentFeatures:
    student_id: str
    total_speaking_time: float = 0.0
    num_utterances: int = 0
    num_questions: int = 0
    num_ideas: int = 0
    num_disagreements: int = 0
    num_supports: int = 0
    num_off_task_events: int = 0
    hand_raise_count: int = 0
    address_teacher_count: int = 0
    address_class_count: int = 0

    # Видео-признаки (мультимодальные сигналы внимания и поведения).
    # Эти поля заполняются на основе VideoActivitySegment и покадровых видео-фич:
    # - video_attention_score: 0–1, средняя доля времени, когда ученик смотрит на учителя/доску.
    # - gesture_activity_score: 0–1, относительная активность жестов при объяснениях/ответах.
    # - facial_engagement_score: 0–1, вовлечённость по мимике (интерес/радость vs скука/отключение).
    # - video_off_task_events: количество оффтоп-эпизодов, зафиксированных по видео (болтовня, хождение и т.п.).
    video_attention_score: float = 0.0
    gesture_activity_score: float = 0.0
    facial_engagement_score: float = 0.0
    video_off_task_events: int = 0

    activity_breakdown: Dict[ActivityType, int] = field(
        default_factory=lambda: {
            "silent": 0,
            "listening": 0,
            "asking_question": 0,
            "answering": 0,
            "giving_idea": 0,
            "disagreeing": 0,
            "supporting_peer": 0,
            "off_task": 0,
        }
    )

    def to_dict(self) -> Dict[str, float | int]:
        data: Dict[str, float | int] = {
            "total_speaking_time": self.total_speaking_time,
            "num_utterances": self.num_utterances,
            "num_questions": self.num_questions,
            "num_ideas": self.num_ideas,
            "num_disagreements": self.num_disagreements,
            "num_supports": self.num_supports,
            "num_off_task_events": self.num_off_task_events,
            "hand_raise_count": self.hand_raise_count,
            "address_teacher_count": self.address_teacher_count,
            "address_class_count": self.address_class_count,
            "video_attention_score": self.video_attention_score,
            "gesture_activity_score": self.gesture_activity_score,
            "facial_engagement_score": self.facial_engagement_score,
            "video_off_task_events": self.video_off_task_events,
        }
        for k, v in self.activity_breakdown.items():
            data[f"activity_{k}"] = v
        return data


@dataclass
class AnnotationSchema:
    """
    Описание формата аннотаций для хранения разметки.
    """

    fields: List[Tuple[str, str]]


DEFAULT_ANNOTATION_SCHEMA = AnnotationSchema(
    fields=[
        ("student_id", "str, идентификатор ученика или track_id"),
        ("start", "float, секунда начала интервала"),
        ("end", "float, секунда конца интервала"),
        (
            "activity_type",
            "one of ActivityType, описывает тип активности (вопрос, ответ, идея и т.п.)",
        ),
        ("text", "str, расшифровка реплики (если есть)"),
    ]
)

