from __future__ import annotations

from pathlib import Path
from typing import List

import whisper_timestamped as whisper

from features import ActivityType, Utterance
from .activity_classifier import classify_activity


def _classify_activity(text: str) -> ActivityType:
    """
    Обёртка над обучаемым классификатором активности.

    Оставлена для обратной совместимости, чтобы минимально менять код,
    но фактическая логика реализована в audio_processing.activity_classifier.classify_activity.
    """
    return classify_activity(text)


def transcribe_audio_to_utterances(
    audio_path: str | Path,
    model_size: str = "small",
) -> List[Utterance]:
    """
    Распознаёт речь и возвращает список Utterance.

    Пока без полноценной диаризации: все реплики помечаются как
    student_id=\"unknown\", позже можно заменить на pyannote/whisper diarization.
    """
    audio_path = Path(audio_path)
    model = whisper.load_model(model_size, device="cpu")
    result = whisper.transcribe(model, str(audio_path), language="ru")

    utterances: List[Utterance] = []
    for segment in result.get("segments", []):
        text = segment.get("text", "").strip()
        if not text:
            continue
        start = float(segment.get("start", 0.0))
        end = float(segment.get("end", start))
        activity_type = _classify_activity(text)
        utterances.append(
            Utterance(
                student_id="unknown",
                start=start,
                end=end,
                text=text,
                activity_type=activity_type,
            )
        )
    return utterances

