from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

# Для возврата (frame_index, bbox) по треку
TrackFrameBbox = Tuple[int, Tuple[int, int, int, int]]

import cv2

from audio_processing import extract_audio_to_wav, transcribe_audio_to_utterances
from features import StudentFeatures, Utterance, VideoActivitySegment, aggregate_student_features
from rubric import Rubric, load_rubric
from scoring import StudentScores, score_students, score_students_model_based
from video_processing import iter_frames, track_students_simple
from video_processing.tracking import TrackedDetection
from video_processing.pose_events import build_video_segments_from_tracks
from video_processing.emotion import estimate_facial_engagement_per_track


def analyze_video(
    video_path: str | Path,
    work_dir: str | Path | None = None,
    min_track_segments: int = 30,
) -> tuple[Rubric, Dict[str, StudentFeatures], Dict[str, StudentScores], Dict[str, str], Dict[str, TrackFrameBbox], Dict[str, bytes]]:
    """
    Полный анализ видео:
    - трекинг учеников;
    - извлечение аудио и распознавание речи;
    - агрегация признаков;
    - расчёт уровней по рубрике.
    """
    video_path = Path(video_path)
    work_dir = Path(work_dir) if work_dir is not None else video_path.parent / "data"
    work_dir.mkdir(parents=True, exist_ok=True)

    # Видео: трекинг людей в кадре.
    frame_iter = iter_frames(video_path)
    tracked = track_students_simple(frame_iter)

    # Видео-события: эвристики по трекам (поднятая рука, оффтоп и т.п.).
    video_segments: List[VideoActivitySegment] = build_video_segments_from_tracks(tracked)

    # Аудио и ASR.
    audio_dir = work_dir / "audio"
    audio_path = extract_audio_to_wav(video_path, audio_dir)
    utterances: List[Utterance] = transcribe_audio_to_utterances(audio_path)

    # Пока реплики не привязаны к конкретным ученикам: распределить их
    # равномерно по имеющимся track_id (очень грубая эвристика).
    unique_track_ids = sorted({seg.student_id for seg in video_segments}) or ["unknown"]
    for idx, utt in enumerate(utterances):
        assigned_student = unique_track_ids[idx % len(unique_track_ids)]
        utt.student_id = assigned_student

    student_features: Dict[str, StudentFeatures] = aggregate_student_features(
        utterances=utterances,
        video_segments=video_segments,
    )

    # Оценка вовлечённости по мимике (если доступна модель FER).
    # Используем те же кадры, что и для превью по трекам.
    # Для экономии ресурсов берём по одному кадру на трек.
    track_id_to_frame_bbox: Dict[str, TrackFrameBbox] = {}
    for seg in video_segments:
        # запомним один из кадров для каждого трека
        if seg.student_id not in track_id_to_frame_bbox:
            tid_int = int(seg.student_id) if seg.student_id.isdigit() else None
            if tid_int is not None:
                # найдём соответствующий detection
                for d in tracked:
                    if d.track_id == tid_int:
                        track_id_to_frame_bbox[seg.student_id] = (d.frame_index, d.bbox)
                        break

    # Предзагружаем несколько кадров по сохранённым индексам.
    video_frames: List[Tuple[int, "cv2.Mat"]] = []
    if track_id_to_frame_bbox:
        cap_eng = cv2.VideoCapture(str(video_path.resolve()))
        if cap_eng.isOpened():
            try:
                seen_indices = set()
                for frame_index, _bbox in track_id_to_frame_bbox.values():
                    if frame_index in seen_indices:
                        continue
                    seen_indices.add(frame_index)
                    frame = _read_frame_at(cap_eng, frame_index)
                    if frame is not None:
                        video_frames.append((frame_index, frame))
            finally:
                cap_eng.release()

    engagement_per_track = estimate_facial_engagement_per_track(
        video_frames, track_id_to_frame_bbox
    )

    for sid, sf in student_features.items():
        score = engagement_per_track.get(sid)
        if score is not None:
            sf.facial_engagement_score = score

    # Отсекаем только шумные короткие треки: ученик должен быть виден минимум min_track_segments кадров
    # (при 25 fps это ~1.2 с). Лимита на количество учеников нет — показываем всех, кто прошёл фильтр.
    if min_track_segments > 0:
        segment_counts = Counter(seg.student_id for seg in video_segments)
        student_features = {
            sid: sf for sid, sf in student_features.items()
            if segment_counts.get(sid, 0) >= min_track_segments
        }

    rubric = load_rubric()
    # Пытаемся использовать модельный скоринг, если обученные модели доступны.
    # Если нет — score_students_model_based сам вернётся к rule-based логике.
    student_scores: Dict[str, StudentScores] = score_students_model_based(
        rubric, student_features
    )

    # Имена только из ручного ввода в блоке «Кто есть кто»; автоопределение лиц отключено.
    sorted_ids = sorted(student_features.keys())
    id_to_label = {sid: f"Ученик {i}" for i, sid in enumerate(sorted_ids, 1)}
    features_labeled = {id_to_label[k]: student_features[k] for k in sorted_ids}
    scores_labeled = {id_to_label[k]: student_scores[k] for k in sorted_ids}

    # Превью по трекам: кадр + JPEG-байты, чтобы в UI сразу показывать фото без открытия видео.
    label_to_track_id: Dict[str, str] = {id_to_label[sid]: sid for sid in sorted_ids}
    # используем уже собранные кадры для engagement; при отсутствии — рассчитываем как раньше
    full_track_id_to_frame_bbox: Dict[str, TrackFrameBbox] = dict(track_id_to_frame_bbox)
    for sid in sorted_ids:
        if sid in full_track_id_to_frame_bbox:
            continue
        tid = int(sid) if (isinstance(sid, str) and sid.isdigit()) else None
        if tid is None:
            continue
        dets = [(d.frame_index, d.bbox) for d in tracked if d.track_id == tid]
        if dets:
            mid = len(dets) // 2
            full_track_id_to_frame_bbox[sid] = dets[mid]
    track_id_to_image_bytes: Dict[str, bytes] = _crop_frames_to_jpeg_bytes(
        video_path, list(full_track_id_to_frame_bbox.keys()), tracked
    )
    _save_thumbnails(video_path, work_dir, tracked, list(sorted_ids))

    return (
        rubric,
        features_labeled,
        scores_labeled,
        label_to_track_id,
        full_track_id_to_frame_bbox,
        track_id_to_image_bytes,
    )


def _crop_to_face_or_person(frame: "cv2.Mat", bbox: Tuple[int, int, int, int], pad: int = 20) -> "cv2.Mat":
    """
    Вырезает область для превью: по возможности по лицу, иначе по верхней части человека (голова/плечи).

    Улучшенный алгоритм:
    - сначала ищет лица на всём кадре и выбирает то, чей центр ближе всего к центру bbox трека;
    - если не нашёл — использует старое поведение (верхняя часть bbox).
    """
    x, y, w, h = bbox
    H, W = frame.shape[0], frame.shape[1]

    # 1. Попытаться найти лицо по всему кадру и выбрать ближайшее к треку.
    gray_full = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    faces = face_cascade.detectMultiScale(
        gray_full, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40)
    )
    if len(faces) > 0:
        cx_track = x + w / 2.0
        cy_track = y + h / 3.0  # чуть выше центра, ближе к голове
        best_face = None
        best_dist = None
        for (fx, fy, fw, fh) in faces:
            cx_face = fx + fw / 2.0
            cy_face = fy + fh / 2.0
            dist = (cx_face - cx_track) ** 2 + (cy_face - cy_track) ** 2
            if best_dist is None or dist < best_dist:
                best_dist = dist
                best_face = (fx, fy, fw, fh)
        if best_face is not None:
            fx, fy, fw, fh = best_face
            face_pad = 20
            fx1 = max(0, fx - face_pad)
            fy1 = max(0, fy - face_pad)
            fx2 = min(W, fx + fw + face_pad)
            fy2 = min(H, fy + fh + face_pad)
            crop = frame[fy1:fy2, fx1:fx2]
            if crop.size > 0:
                return crop

    # 2. Фоллбек: вырезаем окрестность around bbox, как раньше.
    x1 = max(0, x - pad)
    y1 = max(0, y - pad)
    x2 = min(W, x + w + pad)
    y2 = min(H, y + h + pad)
    roi = frame[y1:y2, x1:x2]
    if roi.size == 0:
        return frame[y1:y2, x1:x2]

    head_h = max(h // 2, min(120, h))
    y2_head = min(H, y + head_h + pad)
    crop = frame[y1:y2_head, x1:x2]
    return crop if crop.size > 0 else roi


def _read_frame_at(cap: "cv2.VideoCapture", frame_index: int):
    """Читает кадр по индексу. При неудачном seek перематывает по кадрам (надёжно для любых коденов)."""
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ret, frame = cap.read()
    if ret and frame is not None:
        return frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    for _ in range(frame_index + 1):
        ret, frame = cap.read()
        if not ret or frame is None:
            return None
    return frame


def _crop_frames_to_jpeg_bytes(
    video_path: Path,
    track_ids: List[str],
    tracked: List[TrackedDetection],
) -> Dict[str, bytes]:
    """Извлекает по одному кадру на трек и возвращает JPEG-байты для отображения в UI."""
    out: Dict[str, bytes] = {}
    path_str = str(video_path.resolve())
    cap = cv2.VideoCapture(path_str)
    if not cap.isOpened():
        return out
    try:
        for sid in track_ids:
            tid = int(sid) if (isinstance(sid, str) and sid.isdigit()) else None
            if tid is None:
                continue
            dets = [(d.frame_index, d.bbox) for d in tracked if d.track_id == tid]
            if not dets:
                continue
            mid = len(dets) // 2
            frame_index, bbox = dets[mid]
            frame = _read_frame_at(cap, frame_index)
            if frame is None:
                continue
            crop = _crop_to_face_or_person(frame, bbox)
            if crop.size > 0:
                _, jpeg = cv2.imencode(".jpg", crop)
                if jpeg is not None:
                    out[sid] = jpeg.tobytes()
    finally:
        cap.release()
    return out


def _save_thumbnails(
    video_path: Path,
    work_dir: Path,
    tracked: List[TrackedDetection],
    track_ids: List[str],
) -> None:
    """Сохраняет по одному кадру на трек в work_dir/thumbnails/{track_id}.jpg."""
    thumb_dir = (work_dir / "thumbnails").resolve()
    thumb_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path.resolve()))
    if not cap.isOpened():
        return
    try:
        for sid in track_ids:
            tid = int(sid) if (isinstance(sid, str) and sid.isdigit()) else None
            if tid is None:
                continue
            dets = [(d.frame_index, d.bbox) for d in tracked if d.track_id == tid]
            if not dets:
                continue
            mid = len(dets) // 2
            frame_index, bbox = dets[mid]
            frame = _read_frame_at(cap, frame_index)
            if frame is None:
                continue
            crop = _crop_to_face_or_person(frame, bbox)
            if crop.size > 0:
                out_path = thumb_dir / f"{sid}.jpg"
                cv2.imwrite(str(out_path), crop)
    finally:
        cap.release()


def generate_preview_frames(
    video_path: str | Path,
    work_dir: str | Path,
    num_tracks: int = 2,
) -> Dict[str, TrackFrameBbox]:
    """
    Только трекинг + сохранение превью по кадрам. Не делает ASR и скоринг.
    Возвращает track_id_to_frame_bbox с ключами "1", "2", ... для отображения в UI.
    """
    video_path = Path(video_path)
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    frame_iter = iter_frames(video_path)
    tracked = track_students_simple(frame_iter)
    counts: Dict[int, int] = {}
    for d in tracked:
        counts[d.track_id] = counts.get(d.track_id, 0) + 1
    top_tids = sorted(counts.keys(), key=lambda t: counts[t], reverse=True)[:num_tracks]
    track_id_to_frame_bbox: Dict[str, TrackFrameBbox] = {}
    for i, tid in enumerate(top_tids, 1):
        sid = str(i)
        dets = [(d.frame_index, d.bbox) for d in tracked if d.track_id == tid]
        if dets:
            mid = len(dets) // 2
            track_id_to_frame_bbox[sid] = dets[mid]
    _save_thumbnails(video_path, work_dir, tracked, list(track_id_to_frame_bbox.keys()))
    return track_id_to_frame_bbox


__all__ = ["analyze_video", "generate_preview_frames"]

