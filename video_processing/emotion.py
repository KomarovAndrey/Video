from __future__ import annotations

"""
Оценка вовлечённости по мимике лица.

Использует библиотеку `fer`, если она доступна. Если установка не удалась
или модель недоступна, возвращает 0.0 (нейтральную вовлечённость).

Это даёт нам безопасный путь постепенно включать анализ мимики,
не ломая пайплайн на средах без GPU/зависимостей.
"""

from typing import Dict, Iterable, List, Tuple

import numpy as np

try:
    from fer import FER

    _FER_MODEL = FER(mtcnn=True)
except Exception:  # pragma: no cover - graceful degradation
    _FER_MODEL = None


def estimate_facial_engagement_for_crops(
    crops: Iterable["np.ndarray"],
) -> float:
    """
    Возвращает оценку вовлечённости в диапазоне [0, 1] по набору кропов лица.

    1. Для каждого кропа определяем доминирующую эмоцию через FER.
    2. Классы объединяются в категории:
       - engaged: happy, surprise
       - neutral: neutral
       - bored/negative: sad, angry, disgust, fear.
    3. Итоговая метрика: доля engaged среди всех кропов,
       при сильном преобладании bored/negative может снижаться.
    """
    if _FER_MODEL is None:
        return 0.0

    emo_counts = {"engaged": 0, "neutral": 0, "bored_negative": 0}

    for crop in crops:
        if crop is None or getattr(crop, "size", 0) == 0:
            continue
        try:
            top = _FER_MODEL.top_emotion(crop)
        except Exception:
            continue
        if not top or len(top) != 2:
            continue
        label, _score = top
        label = (label or "").lower()
        if label in ("happy", "surprise"):
            emo_counts["engaged"] += 1
        elif label in ("neutral",):
            emo_counts["neutral"] += 1
        elif label in ("sad", "angry", "disgust", "fear"):
            emo_counts["bored_negative"] += 1

    total = sum(emo_counts.values())
    if total == 0:
        return 0.0

    engaged = emo_counts["engaged"]
    bored_negative = emo_counts["bored_negative"]

    base = engaged / total
    penalty = bored_negative / total
    score = float(np.clip(base - 0.5 * penalty, 0.0, 1.0))
    return score


def estimate_facial_engagement_per_track(
    video_frames: List[Tuple[int, "np.ndarray"]],
    frame_bboxes_per_track: Dict[str, Tuple[int, Tuple[int, int, int, int]]],
) -> Dict[str, float]:
    """
    Оценивает вовлечённость по мимике для каждого track_id.

    video_frames: список (frame_index, frame_bgr)
    frame_bboxes_per_track: track_id -> (frame_index, bbox)
    """
    if not frame_bboxes_per_track:
        return {}
    if _FER_MODEL is None:
        return {tid: 0.0 for tid in frame_bboxes_per_track.keys()}

    frame_dict = {idx: frame for idx, frame in video_frames}
    results: Dict[str, float] = {}
    for tid, (frame_index, bbox) in frame_bboxes_per_track.items():
        frame = frame_dict.get(frame_index)
        if frame is None:
            results[tid] = 0.0
            continue
        x, y, w, h = bbox
        H, W = frame.shape[:2]
        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(W, x + w)
        y2 = min(H, y + h)
        crop = frame[y1:y2, x1:x2]
        score = estimate_facial_engagement_for_crops([crop])
        results[tid] = score
    return results


__all__ = ["estimate_facial_engagement_for_crops", "estimate_facial_engagement_per_track"]

