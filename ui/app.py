from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import streamlit as st

import cv2

from analyze import analyze_video, generate_preview_frames
from rubric import load_rubric


def main() -> None:
    st.set_page_config(page_title="Оценка урока по видео", layout="wide")
    st.title("Анализ видеозаписи урока")
    st.markdown(
        "Загрузите видео урока, чтобы получить предварительную оценку учеников "
        "по критериям: **Лидерство, Коммуникация, Саморефлексия, Критическое мышление, Самоконтроль**."
    )

    uploaded = st.file_uploader("Видео файла урока", type=["mp4", "mov", "mkv", "avi"])
    if not uploaded:
        st.info("Пожалуйста, выберите видеофайл.")
        return

    work_dir = Path("data")
    work_dir.mkdir(exist_ok=True)
    (work_dir / "references").mkdir(exist_ok=True)
    video_path = work_dir / uploaded.name
    with video_path.open("wb") as f:
        f.write(uploaded.read())

    # При смене файла сбрасываем старый результат.
    if st.session_state.get("last_video_name") != uploaded.name:
        if "analysis_result" in st.session_state:
            del st.session_state["analysis_result"]
        st.session_state["last_video_name"] = uploaded.name

    # Анализ запускается только по нажатию кнопки.
    run_clicked = st.button("Запустить анализ")
    if run_clicked:
        with st.spinner("Идёт анализ видео, это может занять несколько минут..."):
            try:
                result = analyze_video(video_path, work_dir=work_dir)
            except Exception as e:
                st.error(f"Ошибка анализа: {e}")
                import traceback
                st.code(traceback.format_exc(), language="text")
                return
        if len(result) == 3:
            rubric, features_by_student, scores_by_student = result
            label_to_track_id = {
                label: str(i) for i, label in enumerate(sorted(features_by_student.keys()), 1)
            }
            track_id_to_frame_bbox = {}
            track_id_to_image_bytes = {}
        elif len(result) == 4:
            rubric, features_by_student, scores_by_student, label_to_track_id = result
            track_id_to_frame_bbox = {}
            track_id_to_image_bytes = {}
        elif len(result) == 5:
            rubric, features_by_student, scores_by_student, label_to_track_id, track_id_to_frame_bbox = result
            track_id_to_image_bytes = {}
        else:
            rubric, features_by_student, scores_by_student, label_to_track_id, track_id_to_frame_bbox, track_id_to_image_bytes = result
        st.session_state["analysis_result"] = (
            rubric,
            features_by_student,
            scores_by_student,
            label_to_track_id,
            track_id_to_frame_bbox,
            track_id_to_image_bytes,
        )
        st.session_state["last_video_path"] = str(video_path.resolve())
        if "name_mapping" in st.session_state:
            del st.session_state["name_mapping"]

    if "analysis_result" not in st.session_state:
        st.info("Видео загружено. Нажмите **«Запустить анализ»**, чтобы выполнить расчёт.")
        return

    result = st.session_state["analysis_result"]
    if len(result) == 3:
        rubric, features_by_student, scores_by_student = result
        label_to_track_id = {
            label: str(i) for i, label in enumerate(sorted(features_by_student.keys()), 1)
        }
        track_id_to_frame_bbox = {}
        track_id_to_image_bytes = {}
    elif len(result) == 4:
        rubric, features_by_student, scores_by_student, label_to_track_id = result
        track_id_to_frame_bbox = {}
        track_id_to_image_bytes = {}
    elif len(result) == 5:
        rubric, features_by_student, scores_by_student, label_to_track_id, track_id_to_frame_bbox = result
        track_id_to_image_bytes = {}
    else:
        rubric, features_by_student, scores_by_student, label_to_track_id, track_id_to_frame_bbox, track_id_to_image_bytes = result

    video_path_str = st.session_state.get("last_video_path") or str(video_path.resolve())
    video_path_abs = str(Path(video_path_str).resolve())

    # Попробуем подготовить запасной кадр из середины видео
    fallback_rgb = None
    cap_fallback = cv2.VideoCapture(video_path_abs)
    if cap_fallback.isOpened():
        frame_count = int(cap_fallback.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        mid_index = frame_count // 2 if frame_count > 0 else 0
        cap_fallback.set(cv2.CAP_PROP_POS_FRAMES, mid_index)
        ret_fb, frame_fb = cap_fallback.read()
        if ret_fb and frame_fb is not None:
            fallback_rgb = cv2.cvtColor(frame_fb, cv2.COLOR_BGR2RGB)
    cap_fallback.release()

    # Кто есть кто: превью + ввод имени
    if "name_mapping" not in st.session_state:
        st.session_state["name_mapping"] = {}
    name_mapping = st.session_state["name_mapping"]
    thumb_dir = (work_dir / "thumbnails").resolve()
    thumb_dir.mkdir(parents=True, exist_ok=True)
    st.subheader("Кто есть кто")
    st.caption("Посмотрите на фото и впишите имя человека. Имена появятся в таблице ниже.")
    cols = st.columns(min(len(features_by_student), 5))
    for idx, label in enumerate(sorted(features_by_student.keys())):
        col = cols[idx % len(cols)]
        with col:
            track_id = label_to_track_id.get(label)
            shown = False
            if track_id and track_id in track_id_to_image_bytes:
                st.image(track_id_to_image_bytes[track_id], caption=label, width=220)
                shown = True
            if not shown and track_id:
                thumb_path = thumb_dir / f"{track_id}.jpg"
                if thumb_path.exists():
                    st.image(str(thumb_path), caption=label, width=220)
                    shown = True
            if not shown and track_id and track_id in track_id_to_frame_bbox:
                frame_index, bbox = track_id_to_frame_bbox[track_id]
                cap = cv2.VideoCapture(video_path_abs)
                if cap.isOpened():
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                    ret, frame = cap.read()
                    cap.release()
                    if ret and frame is not None:
                        x, y, w, h = bbox
                        pad = 20
                        x1 = max(0, x - pad)
                        y1 = max(0, y - pad)
                        x2 = min(frame.shape[1], x + w + pad)
                        y2 = min(frame.shape[0], y + h + pad)
                        crop = frame[y1:y2, x1:x2]
                        if crop.size > 0:
                            rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                            st.image(rgb, caption=label, width=220)
                            shown = True
            if not shown:
                if fallback_rgb is not None:
                    st.image(fallback_rgb, caption=label, width=220)
                    shown = True
                else:
                    st.markdown(f"**{label}**")
            new_name = st.text_input(
                f"Имя для «{label}»",
                value=name_mapping.get(label, ""),
                key=f"name_{label}",
            )
            if new_name.strip():
                name_mapping[label] = new_name.strip()

    # Таблица с уровнями (с подставленными именами, если введены)
    display_names = {label: name_mapping.get(label, label) for label in scores_by_student}
    rows = []
    for student_id, scores in scores_by_student.items():
        row = {"Ученик": display_names.get(student_id, student_id)}
        for criterion in rubric.criteria:
            row[criterion.name] = scores.levels.get(criterion.id, 0)
        rows.append(row)

    if rows:
        df = pd.DataFrame(rows)
        df.index = range(1, len(df) + 1)
        df.index.name = "№"
        st.subheader("Уровни по критериям")
        st.dataframe(df, use_container_width=True)

    # Детализация по ученику (по отображаемому имени)
    student_ids = sorted(features_by_student.keys())
    if student_ids:
        options = [display_names.get(sid, sid) for sid in student_ids]
        selected_display = st.selectbox("Выберите ученика", options)
        selected = next(sid for sid in student_ids if display_names.get(sid, sid) == selected_display)
        feats = features_by_student[selected]
        st.markdown("**Агрегированные признаки**")
        st.json(feats.to_dict())


if __name__ == "__main__":
    main()

