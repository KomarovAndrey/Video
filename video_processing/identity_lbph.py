"""
Слияние треков по лицу через LBPH (OpenCV contrib). Без dlib/face_recognition.

Один человек может получить несколько track_id от трекера. По LBPH объединяем
такие треки в один identity_id, чтобы не дублировать учеников в отчёте.
Работает на Streamlit Cloud (нет зависимости от dlib).
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from .face_match import _crop_face_region, _get_face_cascade
from .tracking import TrackedDetection

try:
    cv2.face.LBPHFaceRecognizer_create()
    _HAS_LBPH = True
except AttributeError:
    _HAS_LBPH = False

# Порог confidence LBPH: ниже = один человек (сливаем треки).
LBPH_MERGE_THRESHOLD = 70
# Сколько кадров максимум брать с одного трека для обучения/предсказания.
MAX_FACES_PER_TRACK = 10
# Минимальный размер лица для обучения.
MIN_FACE_SIZE = 30


def _collect_face_samples(
    video_path: Path,
    tracked: List[TrackedDetection],
    max_per_track: int = MAX_FACES_PER_TRACK,
) -> List[Tuple[int, np.ndarray]]:
    """
    По каждому треку собирает до max_per_track кропов лиц (серые, после Haar).
    Возвращает список (track_id, gray_face_roi).
    """
    cascade = _get_face_cascade()
    if cascade is None:
        return []

    by_track: Dict[int, List[Tuple[int, Tuple[int, int, int, int]]]] = defaultdict(list)
    for d in tracked:
        by_track[d.track_id].append((d.frame_index, d.bbox))

    result: List[Tuple[int, np.ndarray]] = []
    cap = cv2.VideoCapture(str(video_path.resolve()))
    if not cap.isOpened():
        return []

    try:
        for track_id, dets in by_track.items():
            dets_sorted = sorted(dets, key=lambda x: x[0])
            step = max(1, len(dets_sorted) // max_per_track)
            to_use = dets_sorted[::step][:max_per_track]
            for frame_index, bbox in to_use:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                ret, frame = cap.read()
                if not ret or frame is None:
                    continue
                gray_crop = _crop_face_region(frame, bbox)
                if gray_crop is None:
                    continue
                faces = cascade.detectMultiScale(
                    gray_crop,
                    scaleFactor=1.1,
                    minNeighbors=5,
                    minSize=(MIN_FACE_SIZE, MIN_FACE_SIZE),
                )
                for (fx, fy, fw, fh) in faces:
                    face_roi = gray_crop[fy : fy + fh, fx : fx + fw]
                    if face_roi.size > 0:
                        result.append((track_id, face_roi))
                        break  # один кадр — одно лицо на трек
    finally:
        cap.release()

    return result


def _build_track_groups(
    samples: List[Tuple[int, np.ndarray]],
    recognizer: "cv2.face.LBPHFaceRecognizer",
    merge_threshold: float = LBPH_MERGE_THRESHOLD,
) -> Dict[int, int]:
    """
    По предсказаниям LBPH строит группы треков (union-find).
    Возвращает отображение track_id -> representative (min track_id в группе).
    """
    if not samples:
        return {}

    # representative для каждого track_id (union-find parent).
    parent: Dict[int, int] = {}
    for track_id, _ in samples:
        if track_id not in parent:
            parent[track_id] = track_id

    def find(x: int) -> int:
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = min(ra, rb)
            parent[rb] = parent[ra]

    # По каждому сэмплу: предсказание. Если предсказанный label != track_id и confidence низкий — один человек.
    for track_id, face_roi in samples:
        try:
            pred_label, confidence = recognizer.predict(face_roi)
        except Exception:
            continue
        pred_tid = int(pred_label)
        if pred_tid != track_id and confidence < merge_threshold:
            union(track_id, pred_tid)

    # Нормализуем: каждый track_id -> его representative.
    return {tid: find(tid) for tid in parent}


def get_track_id_to_identity_lbph(
    video_path: str | Path,
    tracked: List[TrackedDetection],
    merge_threshold: float = LBPH_MERGE_THRESHOLD,
    max_faces_per_track: int = MAX_FACES_PER_TRACK,
) -> Dict[int, str]:
    """
    Собирает кропы лиц по трекам, обучает LBPH, объединяет треки одного человека.
    Возвращает track_id -> identity_id ("1", "2", ...). Без dlib.
    """
    video_path = Path(video_path)
    unique_track_ids = sorted({d.track_id for d in tracked})
    if not unique_track_ids:
        return {}

    if not _HAS_LBPH:
        return {tid: str(tid) for tid in unique_track_ids}

    samples = _collect_face_samples(video_path, tracked, max_per_track=max_faces_per_track)
    if not samples:
        return {tid: str(tid) for tid in unique_track_ids}

    # Обучаем LBPH: метка = track_id (целое).
    track_ids = [t for t, _ in samples]
    face_images = [img for _, img in samples]
    labels = np.array(track_ids, dtype=np.int32)
    try:
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        recognizer.setThreshold(merge_threshold)
        recognizer.train(face_images, labels)
    except Exception:
        return {tid: str(tid) for tid in unique_track_ids}

    groups = _build_track_groups(samples, recognizer, merge_threshold=merge_threshold)
    # representative -> порядковый identity_id "1", "2", ... (включая треки без лиц)
    reps = sorted(
        set(groups.values()) | {tid for tid in unique_track_ids if tid not in groups}
    )
    rep_to_identity: Dict[int, str] = {rep: str(i) for i, rep in enumerate(reps, 1)}
    return {
        tid: rep_to_identity[groups.get(tid, tid)]
        for tid in unique_track_ids
    }


__all__ = ["get_track_id_to_identity_lbph"]
