from __future__ import annotations

from pathlib import Path

import ffmpeg


def extract_audio_to_wav(video_path: str | Path, output_dir: str | Path) -> Path:
    """
    Извлекает аудиодорожку из видео в формате WAV.
    """
    video_path = Path(video_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    audio_path = output_dir / (video_path.stem + ".wav")

    (
        ffmpeg.input(str(video_path))
        .output(str(audio_path), acodec="pcm_s16le", ac=1, ar="16000")
        .overwrite_output()
        .run(quiet=True)
    )
    return audio_path

