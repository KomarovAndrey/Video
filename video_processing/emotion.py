from __future__ import annotations

"""
Оценка вовлечённости по мимике лица и направлению головы.

Использует библиотеку `fer`, если она доступна. Опционально учитывается
head pose (MediaPipe) для объединения FER + внимание по взгляду.
Если установка не удалась или модель недоступна, возвращается 0.0 (нейтральная вовлечённость).
"""

from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import cv2
import numpy as np

try:
    from fer import FER

    _FER_MODEL = FER(mtcnn=True)
except Exception:  # pragma: no cover - graceful degradation
    _FER_MODEL = None

try:
    from .head_pose import estimate_head_attention_score as _head_attention
except Exception:
    _head_attention = None


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


def estimate_facial_engagement_for_crops_with_head_pose(
    crops: Iterable["np.ndarray"],
    head_pose_weight: float = 0.4,
) -> float:
    """
    Вовлечённость по мимике (FER) и направлению головы.
    Итог: (1 - head_pose_weight) * FER + head_pose_weight * mean(head_attention).
    """
    crops_list = [c for c in crops if c is not None and getattr(c, "size", 0) > 0]
    if not crops_list:
        return 0.0
    fer_score = estimate_facial_engagement_for_crops(crops_list)
    if _head_attention is None:
        return fer_score
    head_scores = [_head_attention(c) for c in crops_list]
    head_avg = sum(head_scores) / len(head_scores) if head_scores else 0.5
    return float(
        np.clip(
            (1.0 - head_pose_weight) * fer_score + head_pose_weight * head_avg,
            0.0,
            1.0,
        )
    )


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


def estimate_facial_engagement_per_identity(
    video_path: str | Path,
    identity_to_frame_bboxes: Dict[str, List[Tuple[int, Tuple[int, int, int, int]]]],
    max_frames_per_identity: int = 15,
    head_pose_weight: float = 0.4,
) -> Dict[str, float]:
    """
    Многокадровая оценка вовлечённости по мимике и head pose для каждого identity.
    По каждому ученику берётся до max_frames_per_identity кадров по треку, затем
    агрегируются FER и head pose.
    """
    if not identity_to_frame_bboxes:
        return {}
    video_path = Path(video_path)
    cap = cv2.VideoCapture(str(video_path.resolve()))
    if not cap.isOpened():
        return {iid: 0.0 for iid in identity_to_frame_bboxes}

    results: Dict[str, float] = {}
    try:
        for identity_id, frame_bbox_list in identity_to_frame_bboxes.items():
            if not frame_bbox_list:
                results[identity_id] = 0.0
                continue
            # Ограничиваем число кадров и равномерно выбираем по времени.
            step = max(1, len(frame_bbox_list) // max_frames_per_identity)
            selected = frame_bbox_list[::step][:max_frames_per_identity]
            crops: List[np.ndarray] = []
            seen_frames: Dict[int, np.ndarray] = {}
            for frame_index, bbox in selected:
                if frame_index not in seen_frames:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                    ret, frame = cap.read()
                    if not ret or frame is None:
                        continue
                    seen_frames[frame_index] = frame
                frame = seen_frames.get(frame_index)
                if frame is None:
                    continue
                x, y, w, h = bbox
                H, W = frame.shape[:2]
                x1, y1 = max(0, x), max(0, y)
                x2, y2 = min(W, x + w), min(H, y + h)
                crop = frame[y1:y2, x1:x2]
                if crop.size > 0 and crop.shape[0] >= 32 and crop.shape[1] >= 32:
                    crops.append(crop)
            if not crops:
                results[identity_id] = 0.0
                continue
            results[identity_id] = estimate_facial_engagement_for_crops_with_head_pose(
                crops, head_pose_weight=head_pose_weight
            )
    finally:
        cap.release()
    return results


__all__ = [
    "estimate_facial_engagement_for_crops",
    "estimate_facial_engagement_for_crops_with_head_pose",
    "estimate_facial_engagement_per_track",
    "estimate_facial_engagement_per_identity",
]

