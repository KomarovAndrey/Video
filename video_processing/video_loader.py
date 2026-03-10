from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Generator, Tuple

import cv2


@dataclass
class VideoMetadata:
    path: Path
    fps: float
    frame_count: int
    width: int
    height: int


def iter_frames(video_path: str | Path) -> Generator[Tuple[int, float, "cv2.Mat"], None, VideoMetadata]:
    """
    Итератор по кадрам видео.

    Возвращает (frame_index, timestamp_seconds, frame_bgr),
    а после завершения — VideoMetadata через `StopIteration.value`.
    """
    path = Path(video_path)
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Не удалось открыть видео: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

    idx = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            timestamp = idx / fps
            yield idx, timestamp, frame
            idx += 1
    finally:
        cap.release()

    return VideoMetadata(path=path, fps=fps, frame_count=frame_count, width=width, height=height)

