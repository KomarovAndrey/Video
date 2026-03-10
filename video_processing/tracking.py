from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import cv2
import numpy as np


@dataclass
class TrackedDetection:
    track_id: int
    frame_index: int
    timestamp: float
    bbox: Tuple[int, int, int, int]  # x, y, w, h


def _detect_people(frame_bgr: "cv2.Mat") -> List[Tuple[int, int, int, int]]:
    """
    Простая детекция людей с помощью встроенного HOG+SVM из OpenCV.
    Для MVP, затем можно заменить на YOLO/другую модель.
    """
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    rects, _ = hog.detectMultiScale(frame_bgr, winStride=(8, 8))
    return [(int(x), int(y), int(w), int(h)) for (x, y, w, h) in rects]


def _centroid(bbox: Tuple[int, int, int, int]) -> Tuple[float, float]:
    x, y, w, h = bbox
    return x + w / 2.0, y + h / 2.0


def track_students_simple(
    frames: Iterable[Tuple[int, float, "cv2.Mat"]],
    max_distance: float = 80.0,
    min_detections_per_track: int = 5,
) -> List[TrackedDetection]:
    """
    Простейший трекер по центроидам:
    - На каждом кадре детектирует людей.
    - Сопоставляет детекции с существующими треками по евклидовой дистанции центров.
    """
    next_track_id = 1
    active_tracks: Dict[int, Tuple[float, float]] = {}
    results: List[TrackedDetection] = []

    for frame_index, timestamp, frame in frames:
        detections = _detect_people(frame)
        used_track_ids: set[int] = set()

        for bbox in detections:
            cx, cy = _centroid(bbox)
            best_id = None
            best_dist = max_distance
            for track_id, (px, py) in active_tracks.items():
                dist = float(np.hypot(cx - px, cy - py))
                if dist < best_dist and track_id not in used_track_ids:
                    best_dist = dist
                    best_id = track_id

            if best_id is None:
                best_id = next_track_id
                next_track_id += 1

            active_tracks[best_id] = (cx, cy)
            used_track_ids.add(best_id)
            results.append(
                TrackedDetection(
                    track_id=best_id,
                    frame_index=frame_index,
                    timestamp=timestamp,
                    bbox=bbox,
                )
            )

    # Отфильтровать "случайные" треки, которые появились на 1–2 кадра,
    # чтобы один ребёнок не превращался в несколько учеников.
    counts: Dict[int, int] = {}
    for det in results:
        counts[det.track_id] = counts.get(det.track_id, 0) + 1

    filtered: List[TrackedDetection] = [
        det for det in results if counts.get(det.track_id, 0) >= min_detections_per_track
    ]

    # Перенумеруем track_id подряд (1, 2, 3, ...), чтобы таблица учеников была аккуратной.
    id_map: Dict[int, int] = {}
    next_id = 1
    for det in filtered:
        if det.track_id not in id_map:
            id_map[det.track_id] = next_id
            next_id += 1
        det.track_id = id_map[det.track_id]

    return filtered

