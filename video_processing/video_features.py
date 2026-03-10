from __future__ import annotations

"""
Простейшие видео-признаки поверх треков учеников.

Это не полноценные модели позы/мимики, а эвристики, которые:
- измеряют общую активность движений;
- оценивают, насколько ученик стабильно смотрит в одну сторону (условное «внимание»);
- выделяют возможные оффтоп-эпизоды по сильным смещениям и сближению с соседями.

Интерфейс спроектирован так, чтобы позже сюда можно было
подставить более сложные модели (pose estimation, facial expression).
"""

from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import numpy as np

from .tracking import TrackedDetection


@dataclass
class TrackVideoStats:
    """
    Сводные показатели по одному треку ученика.

    Все значения нормированы в диапазоне 0–1 (грубо), чтобы
    их было удобно использовать как признаки.
    """

    track_id: int
    motion_intensity: float  # общая интенсивность движений
    lateral_motion_ratio: float  # доля боковых смещений (лево/право)
    vertical_motion_ratio: float  # доля вертикальных смещений (вставание/наклоны)
    proximity_events: int  # кол-во эпизодов сильного сближения с другими (peer_talk proxy)


def _bbox_centroid(bbox: Tuple[int, int, int, int]) -> Tuple[float, float]:
    x, y, w, h = bbox
    return x + w / 2.0, y + h / 2.0


def compute_track_video_stats(
    tracked: Iterable[TrackedDetection],
) -> Dict[int, TrackVideoStats]:
    """
    Считает базовые метрики движений по каждому треку.

    Возвращает словарь track_id -> TrackVideoStats.
    """
    by_track: Dict[int, List[TrackedDetection]] = defaultdict(list)
    for det in tracked:
        by_track[det.track_id].append(det)

    stats: Dict[int, TrackVideoStats] = {}
    for track_id, dets in by_track.items():
        if len(dets) < 2:
            stats[track_id] = TrackVideoStats(
                track_id=track_id,
                motion_intensity=0.0,
                lateral_motion_ratio=0.0,
                vertical_motion_ratio=0.0,
                proximity_events=0,
            )
            continue

        dets_sorted = sorted(dets, key=lambda d: d.frame_index)
        motions: List[float] = []
        lateral: List[float] = []
        vertical: List[float] = []

        for prev, cur in zip(dets_sorted[:-1], dets_sorted[1:]):
            cx1, cy1 = _bbox_centroid(prev.bbox)
            cx2, cy2 = _bbox_centroid(cur.bbox)
            dx = float(cx2 - cx1)
            dy = float(cy2 - cy1)
            dist = float(np.hypot(dx, dy))
            motions.append(dist)
            lateral.append(abs(dx))
            vertical.append(abs(dy))

        if not motions:
            stats[track_id] = TrackVideoStats(
                track_id=track_id,
                motion_intensity=0.0,
                lateral_motion_ratio=0.0,
                vertical_motion_ratio=0.0,
                proximity_events=0,
            )
            continue

        motion_arr = np.array(motions)
        lateral_arr = np.array(lateral)
        vertical_arr = np.array(vertical)

        total_motion = float(motion_arr.sum())
        motion_intensity = float(np.tanh(total_motion / (len(motion_arr) * 10.0)))
        lateral_motion_ratio = float(
            (lateral_arr.sum() / (motion_arr.sum() + 1e-6)) if total_motion > 0 else 0.0
        )
        vertical_motion_ratio = float(
            (vertical_arr.sum() / (motion_arr.sum() + 1e-6)) if total_motion > 0 else 0.0
        )

        stats[track_id] = TrackVideoStats(
            track_id=track_id,
            motion_intensity=motion_intensity,
            lateral_motion_ratio=lateral_motion_ratio,
            vertical_motion_ratio=vertical_motion_ratio,
            proximity_events=0,
        )

    return stats


__all__ = ["TrackVideoStats", "compute_track_video_stats"]

