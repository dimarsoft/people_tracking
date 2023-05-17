from pathlib import Path

import cv2

from tools.labeltools import DetectedTrackLabel, draw_label_text


def draw_on_frame(frame,
                  frame_w: int,
                  frame_h: int,
                  frame_info: DetectedTrackLabel):
    """
    Рисование на фрейме детекции
    Parameters
    ----------
    frame:
        Фрейм.
    frame_w:
        Ширина ббокса.
    frame_h:
        Высота ббокса.
    frame_info
        Результат дедекции

    Returns
    -------

    """
    lab = frame_info

    ww = int(lab.width * frame_w)
    hh = int(lab.height * frame_h)

    x = int(lab.x * frame_w - ww / 2)
    y = int(lab.y * frame_h - hh / 2)

    # рамка объекта

    label_color = (255, 0, 0)  # object_colors(lab.label, True)
    line_width = 2

    cv2.rectangle(frame, (x, y), (x + ww, y + hh), label_color, line_width)

    caption = f"{lab.conf:.2f}"

    draw_label_text(frame, (x, y), caption, line_width, color=(0, 0, 0))


def create_turniket_video(track_labels: list, source_video, output_folder, max_frames: int = -1):
    input_video = cv2.VideoCapture(str(source_video))

    fps = int(input_video.get(cv2.CAP_PROP_FPS))
    # ширина
    w = int(input_video.get(cv2.CAP_PROP_FRAME_WIDTH))
    # высота
    h = int(input_video.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # количество кадров в видео
    frames_in_video = int(input_video.get(cv2.CAP_PROP_FRAME_COUNT))

    results = []

    if max_frames > 0:
        frames_in_video = min(frames_in_video, max_frames)

    for frame_id in range(frames_in_video):
        ret, frame = input_video.read()
        results.append(frame)
    input_video.release()

    for label in track_labels:
        draw_on_frame(results[int(label.frame)], w, h, label)

    output_video_path = str(Path(output_folder) / Path(source_video).name)
    output_video = cv2.VideoWriter(
        output_video_path, cv2.VideoWriter_fourcc(*'mp4v'),
        fps, (w, h))
    # запись в выходной файл
    for i in range(frames_in_video):
        output_video.write(results[i])

    output_video.release()

    del results
