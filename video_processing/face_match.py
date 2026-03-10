"""
Распознавание лица по эталонному фото (LBPH, OpenCV contrib).
Если opencv-contrib не установлен, распознавание не выполняется.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from .tracking import TrackedDetection

# LBPH есть только в opencv-contrib-python
try:
    cv2.face.LBPHFaceRecognizer_create()
    _HAS_FACE_MODULE = True
except AttributeError:
    _HAS_FACE_MODULE = False

_FACE_CASCADE: Optional[cv2.CascadeClassifier] = None


def _get_face_cascade() -> Optional[cv2.CascadeClassifier]:
    global _FACE_CASCADE
    if _FACE_CASCADE is None:
        path = getattr(cv2.data, "haarcascades", None) or ""
        if path:
            _FACE_CASCADE = cv2.CascadeClassifier(
                str(Path(path) / "haarcascade_frontalface_default.xml")
            )
    return _FACE_CASCADE


# Имена файлов (без расширения) -> отображаемое имя в отчёте
REF_NAME_MAP = {"andrey": "Андрей", "islam": "Ислам"}


def find_all_references(work_dir: Path) -> List[Tuple[Path, str]]:
    """Ищет все эталоны в work_dir/references/: andrey.*, islam.* и т.д."""
    ref_dir = work_dir / "references"
    if not ref_dir.exists():
        return []
    result: List[Tuple[Path, str]] = []
    seen_names: set[str] = set()
    for ext in ("jpg", "jpeg", "png"):
        for path in ref_dir.glob(f"*.{ext}"):
            stem = path.stem.lower()
            name = REF_NAME_MAP.get(stem)
            if name is not None and name not in seen_names:
                seen_names.add(name)
                result.append((path, name))
    return result


def train_recognizer_from_references(
    refs: List[Tuple[Path, str]],
) -> Optional[Tuple["cv2.face.LBPHFaceRecognizer", Dict[int, str]]]:
    """Обучает LBPH на нескольких эталонах. Возвращает (recognizer, label_to_name) или None."""
    if not _HAS_FACE_MODULE or not refs:
        return None
    cascade = _get_face_cascade()
    if cascade is None:
        return None
    face_images: List[np.ndarray] = []
    label_to_name: Dict[int, str] = {}
    for i, (image_path, name) in enumerate(refs):
        img = cv2.imread(str(image_path))
        if img is None:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        if len(faces) == 0:
            continue
        x, y, w, h = faces[0]
        face_roi = gray[y : y + h, x : x + w]
        face_images.append(face_roi)
        label_to_name[i] = name
    if not face_images:
        return None
    try:
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        recognizer.setThreshold(70)
        labels = np.array(list(range(len(face_images))))
        recognizer.train(face_images, labels)
        return (recognizer, label_to_name)
    except Exception:
        return None


def train_recognizer_from_reference(
    image_path: Path,
) -> Optional["cv2.face.LBPHFaceRecognizer"]:
    """Обучает LBPH на одном эталонном лице (обратная совместимость)."""
    out = train_recognizer_from_references([(image_path, "Андрей")])
    return out[0] if out else None


def _crop_face_region(frame: np.ndarray, bbox: Tuple[int, int, int, int]) -> Optional[np.ndarray]:
    """Область лица — верхняя часть bbox человека."""
    x, y, w, h = bbox
    h_face = max(int(h * 0.5), 60)
    crop = frame[y : y + h_face, x : x + w]
    if crop.size == 0:
        return None
    return cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)


def match_track_to_reference(
    video_path: Path,
    tracked: List[TrackedDetection],
    track_id: int,
    recognizer: "cv2.face.LBPHFaceRecognizer",
    label_to_name: Dict[int, str],
    max_frames: int = 5,
) -> Optional[str]:
    """Проверяет, совпадает ли лицо в треке с одним из эталонов. Возвращает имя или None."""
    cascade = _get_face_cascade()
    if cascade is None or not label_to_name:
        return None
    detections = [(d.frame_index, d.bbox) for d in tracked if d.track_id == track_id]
    if not detections:
        return None
    step = max(1, len(detections) // max_frames)
    to_check = detections[::step][:max_frames]
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    try:
        for frame_index, bbox in to_check:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ret, frame = cap.read()
            if not ret:
                continue
            gray_crop = _crop_face_region(frame, bbox)
            if gray_crop is None:
                continue
            faces = cascade.detectMultiScale(
                gray_crop, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
            )
            for (fx, fy, fw, fh) in faces:
                face_roi = gray_crop[fy : fy + fh, fx : fx + fw]
                try:
                    label, confidence = recognizer.predict(face_roi)
                    if label in label_to_name and confidence < 70:
                        return label_to_name[label]
                except Exception:
                    continue
    finally:
        cap.release()
    return None
