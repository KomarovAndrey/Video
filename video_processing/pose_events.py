from __future__ import annotations

"""
Эвристическое извлечение видео-событий (поднятие руки, оффтоп) по трекам.

Это не полноценная модель позы: мы используем динамику bbox каждого трека,
чтобы приблизительно выделить:
- потенциальные эпизоды поднятия руки (верх bbox уходит выше среднего);
- эпизоды оффтопа (как и раньше — сильные боковые движения).

Интерфейс спроектирован так, чтобы позже сюда можно было подставить
реальную модель позы (MediaPipe / YOLO-pose), не меняя внешнее API.
"""

from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

from features import VideoActivitySegment
from .tracking import TrackedDetection
from .video_features import compute_track_video_stats


def _per_track_detections(
    tracked: Iterable[TrackedDetection],
) -> Dict[int, List[TrackedDetection]]:
    by_track: Dict[int, List[TrackedDetection]] = defaultdict(list)
    for det in tracked:
        by_track[det.track_id].append(det)
    for dets in by_track.values():
        dets.sort(key=lambda d: d.frame_index)
    return by_track


def build_video_segments_from_tracks(
    tracked: List[TrackedDetection],
    track_id_to_identity: Optional[Dict[int, str]] = None,
) -> List[VideoActivitySegment]:
    """
    Строит список VideoActivitySegment на основе треков.

    Если передан track_id_to_identity (track_id -> identity_id), то student_id в сегментах
    задаётся по лицу: несколько треков одного человека получают один identity_id.
    """
    if not tracked:
        return []

    track_stats = compute_track_video_stats(tracked)
    by_track = _per_track_detections(tracked)

    def _student_id(det: TrackedDetection) -> str:
        if track_id_to_identity:
            return track_id_to_identity.get(det.track_id, str(det.track_id))
        return str(det.track_id)

    segments: List[VideoActivitySegment] = []

    for track_id, dets in by_track.items():
        ys = np.array([d.bbox[1] for d in dets], dtype=float)
        hs = np.array([d.bbox[3] for d in dets], dtype=float)
        if len(ys) == 0:
            continue
        median_y = float(np.median(ys))
        median_h = float(np.median(hs)) if len(hs) else 0.0
        y_threshold = median_y - 0.25 * median_h  # выше среднего положения

        stats = track_stats.get(track_id)
        for det in dets:
            x, y, w, h = det.bbox
            is_hand_raised = bool(median_h > 0 and y < y_threshold)
            is_off_task = False
            if stats is not None:
                is_off_task = bool(
                    stats.motion_intensity > 0.7 and stats.lateral_motion_ratio > 0.6
                )
            segments.append(
                VideoActivitySegment(
                    student_id=_student_id(det),
                    start=det.timestamp,
                    end=det.timestamp,
                    is_hand_raised=is_hand_raised,
                    is_addressing_teacher=False,
                    is_addressing_class=False,
                    is_off_task=is_off_task,
                )
            )

    return segments


__all__ = ["build_video_segments_from_tracks"]

