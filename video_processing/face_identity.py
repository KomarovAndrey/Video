"""
Слияние треков по лицу: один человек = один identity_id.

Трекер может присвоить одному человеку несколько track_id (потерял, снова нашёл).
По эмбеддингам лица объединяем такие треки в одного «ученика», чтобы не дублировать
одного человека в отчёте и не терять второго.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from .tracking import TrackedDetection

import cv2
import numpy as np

try:
    import face_recognition

    _HAS_FACE_RECOGNITION = True
except ImportError:
    _HAS_FACE_RECOGNITION = False


# Порог схожести лиц: меньше = строже (одно лицо не сливается с другим).
FACE_MATCH_TOLERANCE = 0.55


def _crop_face_roi(frame_bgr: np.ndarray, bbox: Tuple[int, int, int, int]) -> Optional[np.ndarray]:
    """Вырезает область лица/головы из кадра по bbox человека."""
    x, y, w, h = bbox
    H, W = frame_bgr.shape[:2]
    # верхняя часть bbox — голова/лицо
    h_head = max(int(h * 0.6), 80)
    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(W, x + w)
    y2 = min(H, y + h_head)
    crop = frame_bgr[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    return crop


def _encode_face(crop_bgr: np.ndarray) -> Optional[np.ndarray]:
    """Один эмбеддинг лица из кропа (BGR). Если лицо не найдено — None."""
    if not _HAS_FACE_RECOGNITION:
        return None
    rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    encodings = face_recognition.face_encodings(rgb)
    if not encodings:
        return None
    return encodings[0]


def merge_tracks_by_face(
    track_id_to_crop: Dict[int, np.ndarray],
    tolerance: float = FACE_MATCH_TOLERANCE,
) -> Dict[int, str]:
    """
    По кропам лиц по каждому track_id возвращает отображение track_id -> identity_id (строка).

    Треки с одним и тем же лицом получают один identity_id. Идентификаторы идут "1", "2", ...
    """
    if not track_id_to_crop or not _HAS_FACE_RECOGNITION:
        return {tid: str(tid) for tid in track_id_to_crop}  # без слияния

    track_ids = sorted(track_id_to_crop.keys())
    encodings_by_track: Dict[int, Optional[np.ndarray]] = {}
    for tid in track_ids:
        enc = _encode_face(track_id_to_crop[tid])
        encodings_by_track[tid] = enc

    # Список «известных» личностей: (identity_id, encoding).
    known: List[Tuple[str, np.ndarray]] = []
    track_id_to_identity: Dict[int, str] = {}

    for tid in track_ids:
        enc = encodings_by_track.get(tid)
        if enc is None:
            # Лицо не найдено — свой id, не сливаем с другими.
            new_id = str(len(known) + 1)
            known.append((new_id, None))
            track_id_to_identity[tid] = new_id
            continue
        enc = np.asarray(enc, dtype=np.float64)
        if enc.ndim > 1:
            enc = enc.ravel()
        # Сравниваем с уже известными.
        matched = False
        for identity_id, known_enc in known:
            if known_enc is None:
                continue
            if face_recognition.compare_faces([known_enc], enc, tolerance=tolerance)[0]:
                track_id_to_identity[tid] = identity_id
                matched = True
                break
        if not matched:
            new_id = str(len(known) + 1)
            known.append((new_id, enc))
            track_id_to_identity[tid] = new_id

    # Перенумеруем identity компактно: "1", "2", "3" без пропусков.
    unique_ids = sorted(set(track_id_to_identity.values()), key=lambda x: (len(x), x))
    old_to_new: Dict[str, str] = {old: str(i) for i, old in enumerate(unique_ids, 1)}
    return {tid: old_to_new[track_id_to_identity[tid]] for tid in track_ids}


def get_track_id_to_identity(
    video_path: str | Path,
    tracked: List["TrackedDetection"],
    track_id_to_crop_fn,
) -> Dict[int, str]:
    """
    Собирает по одному кропу на трек, затем вызывает merge_tracks_by_face.

    track_id_to_crop_fn(track_id, frame_bgr, bbox) -> Optional[ndarray] — функция,
    возвращающая кроп для трека (или None).
    """
    path = Path(video_path)
    by_track: Dict[int, List[Tuple[int, Tuple[int, int, int, int]]]] = defaultdict(list)
    for d in tracked:
        by_track[d.track_id].append((d.frame_index, d.bbox))

    track_id_to_crop: Dict[int, np.ndarray] = {}
    cap = cv2.VideoCapture(str(path.resolve()))
    if not cap.isOpened():
        return {tid: str(tid) for tid in by_track}

    try:
        for track_id, dets in by_track.items():
            if not dets:
                continue
            dets_sorted = sorted(dets, key=lambda x: x[0])
            mid = len(dets_sorted) // 2
            frame_index, bbox = dets_sorted[mid]
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ret, frame = cap.read()
            if not ret or frame is None:
                track_id_to_crop[track_id] = np.zeros((10, 10, 3), dtype=np.uint8)
                continue
            crop = track_id_to_crop_fn(track_id, frame, bbox)
            if crop is not None and crop.size > 0:
                track_id_to_crop[track_id] = crop
            else:
                track_id_to_crop[track_id] = np.zeros((10, 10, 3), dtype=np.uint8)
    finally:
        cap.release()

    return merge_tracks_by_face(track_id_to_crop)


__all__ = ["merge_tracks_by_face", "get_track_id_to_identity", "_crop_face_roi"]
