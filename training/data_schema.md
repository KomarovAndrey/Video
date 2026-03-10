# Схема данных для обучения моделей

Этот файл описывает формат данных, которые нужно собрать и разметить
для обучения мультимодальных моделей (аудио и видео), а также скоринга по рубрике.

## 1. Аудио и текст

- Источник: аудио из уроков (`audio_processing.extract_audio_to_wav`) и транскрипты Whisper.
- Для обучения классификатора активности по тексту рекомендуется собрать CSV:

Столбцы:
- `utterance_id` — уникальный идентификатор реплики.
- `student_id` — идентификатор ученика или трека.
- `start` — начало реплики (секунды).
- `end` — конец реплики (секунды).
- `text` — текст реплики.
- `activity_type` — один из ActivityType:
  - `silent`, `listening`, `asking_question`, `answering`,
  - `giving_idea`, `disagreeing`, `supporting_peer`, `off_task`.

Файл: `data/annotations/utterances.csv`.

## 2. Видео-события

Для обучения видео-признаков и проверки оффтопа по видео рекомендуется собрать:

- `video_id` — идентификатор урока/видео.
- `student_id` — идентификатор ученика (соответствует track_id после маппинга).
- `start` / `end` — интервал времени (секунды).
- `event_type` — тип видео-события:
  - `hand_raise`, `address_teacher`, `address_class`,
  - `video_off_task`, `peer_talk`, `high_gesture_activity`,
  - `high_facial_engagement`, `low_facial_engagement`.

Файл: `data/annotations/video_events.csv`.

## 3. Рубричная разметка (таргеты)

Для обучения финальной модели скоринга по рубрике:

- `video_id` — идентификатор урока.
- `student_id` — ученик.
- `leadership` — уровень 1–5.
- `communication` — уровень 1–5.
- `self_reflection` — уровень 1–5.
- `critical_thinking` — уровень 1–5.
- `self_regulation` — уровень 1–5.

Файл: `data/annotations/rubric_scores.csv`.

