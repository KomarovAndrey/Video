"""
Диарзация речи: кто когда говорил.

При установленном pyannote.audio возвращает сегменты (start, end, speaker_id).
При недоступности возвращает пустой список — вызывающий код использует
равномерное распределение реплик по ученикам (round-robin).
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from features import Utterance

try:
    from pyannote.audio import Pipeline

    _HAS_PYANNOTE = True
except Exception:
    _HAS_PYANNOTE = False

_PIPELINE = None


def _load_pipeline():
    global _PIPELINE
    if not _HAS_PYANNOTE or _PIPELINE is False:
        return None
    if _PIPELINE is None:
        try:
            # Требует Hugging Face token для pyannote/speaker-diarization-3.1
            _PIPELINE = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                use_auth_token=None,
            )
        except Exception:
            _PIPELINE = False
    return _PIPELINE if _PIPELINE is not None else None


def diarize_audio(
    audio_path: str | Path,
    min_duration_off: float = 0.3,
    min_duration_on: float = 0.3,
) -> List[Tuple[float, float, str]]:
    """
    Возвращает список сегментов (start_sec, end_sec, speaker_id).
    speaker_id — строка ("SPEAKER_00", "SPEAKER_01", ...).
    При недоступности pyannote возвращает [].
    """
    audio_path = Path(audio_path)
    if not audio_path.exists():
        return []
    pipeline = _load_pipeline()
    if pipeline is None:
        return []
    try:
        diar = pipeline(str(audio_path), min_duration_off=min_duration_off, min_duration_on=min_duration_on)
        segments: List[Tuple[float, float, str]] = []
        for turn, _, speaker in diar.itertracks(yield_label=True):
            segments.append((float(turn.start), float(turn.end), str(speaker)))
        return segments
    except Exception:
        return []


def assign_utterances_to_speakers(
    utterances: List["Utterance"],
    diar_segments: List[Tuple[float, float, str]],
) -> None:
    """
    Назначает каждому utterance student_id = speaker_id по перекрытию по времени.
    Модифицирует utterance.student_id на месте. Если диарных сегментов нет — не меняет.
    """
    if not diar_segments:
        return
    for utt in utterances:
        mid = (utt.start + utt.end) / 2.0
        best_speaker: str | None = None
        best_overlap = 0.0
        for seg_start, seg_end, speaker in diar_segments:
            overlap = min(utt.end, seg_end) - max(utt.start, seg_start)
            if overlap > best_overlap and seg_start <= mid <= seg_end:
                best_overlap = overlap
                best_speaker = speaker
        if best_speaker is None:
            for seg_start, seg_end, speaker in diar_segments:
                overlap = min(utt.end, seg_end) - max(utt.start, seg_start)
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_speaker = speaker
        if best_speaker is not None:
            utt.student_id = best_speaker


def speaker_order_from_segments(
    diar_segments: List[Tuple[float, float, str]],
) -> List[str]:
    """Уникальный список спикеров в порядке первого появления в сегментах."""
    order: List[str] = []
    seen: set[str] = set()
    for _start, _end, speaker in diar_segments:
        if speaker not in seen:
            seen.add(speaker)
            order.append(speaker)
    return order


def map_speakers_to_identities(
    speaker_order: List[str],
    identity_ids: List[str],
) -> dict[str, str]:
    """
    Сопоставление speaker_id -> identity_id: speaker_order[i] -> identity_ids[i % len(identity_ids)].
    """
    if not identity_ids:
        return {}
    return {
        speaker: identity_ids[i % len(identity_ids)]
        for i, speaker in enumerate(speaker_order)
    }


__all__ = [
    "diarize_audio",
    "assign_utterances_to_speakers",
    "speaker_order_from_segments",
    "map_speakers_to_identities",
]
