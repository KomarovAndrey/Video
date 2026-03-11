"""
Слияние треков по лицу через эмбеддинги InsightFace (ArcFace).

При установленном insightface даёт устойчивое к ракурсу и освещению
слияние треков. Опционально: сравнение с эталонами из work_dir/references/
для именованных identity_id. При недоступности возвращает None — вызывающий
код использует LBPH fallback.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from .tracking import TrackedDetection

try:
    from insightface.app import FaceAnalysis

    _HAS_INSIGHTFACE = True
except Exception:
    _HAS_INSIGHTFACE = False

# Порог косинусной близости: выше = один человек (сливаем треки).
INSIGHTFACE_MERGE_THRESHOLD = 0.45
# Сколько кадров максимум брать с одного трека для эмбеддингов.
MAX_FACES_PER_TRACK = 10
# Минимальная сторона кропа для InsightFace.
MIN_CROP_SIDE = 32

# Ленивая инициализация приложения (один раз на процесс).
_FACE_APP: Optional["FaceAnalysis"] = None


def _get_face_app() -> Optional["FaceAnalysis"]:
    global _FACE_APP
    if not _HAS_INSIGHTFACE:
        return None
    if _FACE_APP is None:
        try:
            _FACE_APP = FaceAnalysis(name="buffalo_l")
            _FACE_APP.prepare(ctx_id=-1, det_size=(320, 320))
        except Exception:
            _FACE_APP = False  # type: ignore[assignment]
    return _FACE_APP if _FACE_APP is not None else None


def _crop_face_region_bgr(
    frame: np.ndarray, bbox: Tuple[int, int, int, int]
) -> Optional[np.ndarray]:
    """Верхняя часть bbox человека (голова/лицо), BGR для InsightFace."""
    x, y, w, h = bbox
    h_face = max(int(h * 0.5), 60)
    H, W = frame.shape[:2]
    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(W, x + w)
    y2 = min(H, y + h_face)
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0 or crop.shape[0] < MIN_CROP_SIDE or crop.shape[1] < MIN_CROP_SIDE:
        return None
    return crop


def _collect_embeddings_per_track(
    video_path: Path,
    tracked: List[TrackedDetection],
    face_app: "FaceAnalysis",
    max_per_track: int = MAX_FACES_PER_TRACK,
) -> List[Tuple[int, np.ndarray]]:
    """
    Собирает по трекам кропы, извлекает эмбеддинги через InsightFace.
    Возвращает список (track_id, embedding).
    """
    by_track: Dict[int, List[Tuple[int, Tuple[int, int, int, int]]]] = defaultdict(list)
    for d in tracked:
        by_track[d.track_id].append((d.frame_index, d.bbox))

    result: List[Tuple[int, np.ndarray]] = []
    cap = cv2.VideoCapture(str(video_path.resolve()))
    if not cap.isOpened():
        return []

    try:
        for track_id, dets in sorted(by_track.items()):
            dets_sorted = sorted(dets, key=lambda x: x[0])
            step = max(1, len(dets_sorted) // max_per_track)
            to_use = dets_sorted[::step][:max_per_track]
            for frame_index, bbox in to_use:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                ret, frame = cap.read()
                if not ret or frame is None:
                    continue
                crop = _crop_face_region_bgr(frame, bbox)
                if crop is None:
                    continue
                faces = face_app.get(crop)
                if not faces:
                    continue
                emb = getattr(faces[0], "embedding", None)
                if emb is not None and hasattr(emb, "shape") and len(emb.shape) >= 1:
                    result.append((track_id, np.asarray(emb, dtype=np.float32)))
                    break
    finally:
        cap.release()

    return result


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    n = (a * a).sum() ** 0.5 * (b * b).sum() ** 0.5
    if n <= 0:
        return 0.0
    return float(np.dot(a, b) / n)


def _build_track_groups_embedding(
    samples: List[Tuple[int, np.ndarray]],
    merge_threshold: float = INSIGHTFACE_MERGE_THRESHOLD,
) -> Dict[int, int]:
    """
    Union-find: сливаем треки, у которых есть хотя бы одна пара эмбеддингов
    с косинусной близостью >= merge_threshold.
    """
    if not samples:
        return {}

    # Один репрезентативный эмбеддинг на трек (средний или первый).
    track_to_embs: Dict[int, List[np.ndarray]] = defaultdict(list)
    for track_id, emb in samples:
        track_to_embs[track_id].append(emb)
    track_to_rep: Dict[int, np.ndarray] = {}
    for tid, embs in track_to_embs.items():
        track_to_rep[tid] = np.mean(embs, axis=0)

    parent: Dict[int, int] = {tid: tid for tid in track_to_rep}

    def find(x: int) -> int:
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = min(ra, rb)
            parent[rb] = parent[ra]

    tids = list(track_to_rep.keys())
    for i, tid_a in enumerate(tids):
        for tid_b in tids[i + 1 :]:
            sim = _cosine_similarity(track_to_rep[tid_a], track_to_rep[tid_b])
            if sim >= merge_threshold:
                union(tid_a, tid_b)

    return {tid: find(tid) for tid in parent}


def _load_reference_embeddings(
    work_dir: Path, face_app: "FaceAnalysis"
) -> List[Tuple[str, np.ndarray]]:
    """Загружает эталоны из work_dir/references/: папки с именами andrey, islam и т.д."""
    ref_dir = work_dir / "references"
    if not ref_dir.exists():
        return []
    result: List[Tuple[str, np.ndarray]] = []
    for subdir in sorted(ref_dir.iterdir()):
        if not subdir.is_dir():
            continue
        name = subdir.name
        for ext in ("jpg", "jpeg", "png"):
            for path in subdir.glob(f"*.{ext}"):
                img = cv2.imread(str(path))
                if img is None:
                    continue
                faces = face_app.get(img)
                if not faces:
                    continue
                emb = getattr(faces[0], "embedding", None)
                if emb is not None:
                    result.append((name, np.asarray(emb, dtype=np.float32)))
                break
            else:
                continue
            break
    return result


def get_track_id_to_identity_insightface(
    video_path: str | Path,
    tracked: List[TrackedDetection],
    work_dir: Optional[str | Path] = None,
    merge_threshold: float = INSIGHTFACE_MERGE_THRESHOLD,
    max_faces_per_track: int = MAX_FACES_PER_TRACK,
) -> Optional[Dict[int, str]]:
    """
    Слияние треков по эмбеддингам InsightFace. Один человек = один identity_id.

    При установленном insightface возвращает Dict[track_id, identity_id].
    identity_id — "1", "2", ... или имя из references/ при совпадении с эталоном.
    При недоступности библиотеки или ошибке возвращает None (использовать LBPH).
    """
    if not _HAS_INSIGHTFACE:
        return None
    face_app = _get_face_app()
    if face_app is None:
        return None

    video_path = Path(video_path)
    work_dir = Path(work_dir) if work_dir is not None else video_path.parent / "data"
    unique_track_ids = sorted({d.track_id for d in tracked})
    if not unique_track_ids:
        return {}

    samples = _collect_embeddings_per_track(
        video_path, tracked, face_app, max_per_track=max_faces_per_track
    )
    if not samples:
        return {tid: str(tid) for tid in unique_track_ids}

    groups = _build_track_groups_embedding(samples, merge_threshold=merge_threshold)
    reps = sorted(
        set(groups.values()) | {tid for tid in unique_track_ids if tid not in groups}
    )

    # Опционально: сопоставить представителей с эталонами.
    ref_embeddings = _load_reference_embeddings(work_dir, face_app)
    rep_to_identity: Dict[int, str] = {}
    if ref_embeddings:
        rep_to_embs: Dict[int, List[np.ndarray]] = defaultdict(list)
        for tid, emb in samples:
            rep = groups.get(tid, tid)
            rep_to_embs[rep].append(emb)
        next_num = 1
        for rep in reps:
            embs = rep_to_embs.get(rep, [])
            rep_emb = np.mean(embs, axis=0) if embs else None
            if rep_emb is None:
                rep_to_identity[rep] = str(next_num)
                next_num += 1
                continue
            best_name: Optional[str] = None
            best_sim = merge_threshold
            for ref_name, ref_emb in ref_embeddings:
                sim = _cosine_similarity(rep_emb, ref_emb)
                if sim > best_sim:
                    best_sim = sim
                    best_name = ref_name
            rep_to_identity[rep] = best_name if best_name else str(next_num)
            if not best_name:
                next_num += 1
    else:
        rep_to_identity = {rep: str(i) for i, rep in enumerate(reps, 1)}

    return {
        tid: rep_to_identity[groups.get(tid, tid)]
        for tid in unique_track_ids
    }


__all__ = ["get_track_id_to_identity_insightface"]
