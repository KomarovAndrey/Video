"""
Оценка внимания по направлению головы (head pose).

Опционально использует MediaPipe Face Mesh для оценки «вперёд/в сторону».
Если библиотека недоступна, возвращает нейтральную оценку 0.5.
Используется вместе с FER для расчёта facial_engagement_score.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

try:
    import cv2
    import mediapipe as mp

    _HAS_MEDIAPIPE = True
except Exception:
    _HAS_MEDIAPIPE = False

# Нос (кончик), левый/правый глаз (внутренние углы) в Face Mesh.
_NOSE_TIP = 1
_LEFT_EYE_INNER = 33
_RIGHT_EYE_INNER = 263
_LEFT_EYE = 468  # нет такого; используем 33 и 263 для горизонтального размаха
# Упрощение: по смещению носа от центра лица грубо оцениваем поворот.
_MAX_NOSE_OFFSET_RATIO = 0.4  # при большем смещении считаем «отвернулся»


def estimate_head_attention_score(crop_bgr: np.ndarray) -> float:
    """
    Оценка «внимательности» по направлению головы: [0, 1].
    Лицо к камере → ближе к 1, сильный поворот в сторону → ближе к 0.
    При недоступности MediaPipe возвращает 0.5 (нейтрально).
    """
    if not _HAS_MEDIAPIPE or crop_bgr is None or getattr(crop_bgr, "size", 0) == 0:
        return 0.5

    try:
        h, w = crop_bgr.shape[:2]
        if h < 32 or w < 32:
            return 0.5
        rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        mp_face_mesh = mp.solutions.face_mesh
        with mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            min_detection_confidence=0.5,
        ) as face_mesh:
            results = face_mesh.process(rgb)
            if not results.multi_face_landmarks:
                return 0.5
            lm = results.multi_face_landmarks[0]
            # Нормализованные координаты (x, y от 0 до 1).
            nose_x = lm.landmark[_NOSE_TIP].x
            nose_y = lm.landmark[_NOSE_TIP].y
            left_x = lm.landmark[_LEFT_EYE_INNER].x
            right_x = lm.landmark[_RIGHT_EYE_INNER].x
            center_x = (left_x + right_x) / 2.0
            # Смещение носа от центра между глазами — грубый признак поворота.
            offset = abs(nose_x - center_x)
            # Чем больше offset, тем меньше «внимание».
            score = float(np.clip(1.0 - offset / _MAX_NOSE_OFFSET_RATIO, 0.0, 1.0))
            return score
    except Exception:
        return 0.5


__all__ = ["estimate_head_attention_score"]
