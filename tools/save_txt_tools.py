"""
Сохраняет результаты детекции YOLO8 и трекинга в текстовый файл

Формат: frame_index track_id class bbox_left bbox_top bbox_w bbox_h conf

bbox - в относительный величинах
"""
import json
from pathlib import Path
from typing import List
import pandas as pd
import numpy as np
from pandas import DataFrame
from ultralytics.engine.results import Results

from tools.exception_tools import print_exception
from tools.track_objects import Track


def convert_txt_toy7(results: list) -> list:
    """
    Конвертор списка детекции
    Parameters
    ----------
    results

    Returns
    -------

    """
    results_y7 = []

    for track in results:
        frame_index = track[0]
        xywhn = track

        bbox_w = xywhn[5]
        bbox_h = xywhn[6]
        bbox_left = xywhn[3]
        bbox_top = xywhn[4]

        track_id = int(track[1])
        cls = int(track[2])

        results_y7.append([frame_index, bbox_left, bbox_top, bbox_w, bbox_h, track_id, cls])

    return results_y7


def convert_toy7(results: List[Results], save_none_id: bool = False, max_frames: int = -1) -> list:
    """
    Конвертация списка детекций в формате ultralytics в общий формат
    Parameters
    ----------
    results:
        Список результатов
    save_none_id:
        Сохранять результат, когда нет ИД трека
    max_frames:
        Максимальное кол-во детекций для сохранения.
        Если -1 все, т.е. нет ограничений

    Returns
    -------
        Возвращает список в общем формате

    """
    results_y7 = []

    for frame_index, track in enumerate(results):
        if 0 < max_frames <= frame_index:
            break
        track = track.cpu()

        if track.boxes is not None:
            boxes = track.boxes
            for box in boxes:
                if save_none_id:
                    track_id = -1 if box.id is None else int(box.id)

                else:
                    if box.id is None:
                        continue
                    track_id = int(box.id)
                # if box.conf < conf:
                #    continue
                # print(frame_index, " ", box.id, ", cls = ", box.cls, ", ", box.xywhn)
                xywhn = box.xywhn.numpy()[0]
                # print(frame_index, " ", xywhn)
                bbox_w = xywhn[2]
                bbox_h = xywhn[3]
                bbox_left = xywhn[0] - bbox_w / 2
                bbox_top = xywhn[1] - bbox_h / 2

                cls = int(box.cls)

                results_y7.append([frame_index, bbox_left, bbox_top,
                                   bbox_w, bbox_h, track_id, cls, box.conf])
    return results_y7


def yolo8_save_tracks_to_txt(results: List[Results], txt_path: str,
                             conf: float = 0.0, save_id: bool = False):
    """

    Args:
        save_id:
        conf: элементы с conf менее указанной не сохраняются
        txt_path: Текстовый файл для сохранения
        results: результат работы модели
    """
    with open(txt_path, 'a') as text_file:
        for frame_index, track in enumerate(results):
            if track.boxes is not None:
                for box in track.boxes:
                    if box.id is None and not save_id:
                        continue
                    if box.conf < conf:
                        continue
                    # print(frame_index, " ", box.id, ", cls = ", box.cls, ", ", box.xywhn)
                    xywhn = box.xywhn.numpy()[0]
                    # print(frame_index, " ", xywhn)
                    bbox_w = xywhn[2]
                    bbox_h = xywhn[3]
                    bbox_left = xywhn[0] - bbox_w / 2
                    bbox_top = xywhn[1] - bbox_h / 2
                    track_id = int(box.id) if box.id is not None else -1
                    cls = int(box.cls)
                    text_file.write(('%g ' * 8 + '\n') % (frame_index, track_id, cls, bbox_left,
                                                          bbox_top, bbox_w, bbox_h, box.conf))


def yolo8_save_detection_to_txt(results: List[Results], txt_path, conf=0.0, save_id=False):
    """

    Args:
        save_id: Сохранять трек ИД?
        conf: элементы с conf менее указанной не сохраняются
        txt_path: Текстовый файл для сохранения
        results: результат работы модели
    """
    with open(txt_path, 'a') as text_file:
        for frame_index, track in enumerate(results):
            if track.boxes is not None:
                for box in track.boxes:
                    if box.id is None and not save_id:
                        continue
                    if box.conf < conf:
                        continue
                    # print(frame_index, " ", box.id, ", cls = ", box.cls, ", ", box.xywhn)
                    xywhn = box.xywhn.numpy()[0]
                    # print(frame_index, " ", xywhn)
                    bbox_w = xywhn[2]
                    bbox_h = xywhn[3]
                    bbox_left = xywhn[0] - bbox_w / 2
                    bbox_top = xywhn[1] - bbox_h / 2
                    track_id = int(box.id) if box.id is not None else -1
                    cls = int(box.cls)
                    text_file.write(('%g ' * 8 + '\n') % (frame_index, track_id, cls, bbox_left,
                                                          bbox_top, bbox_w, bbox_h, box.conf))


def yolo7_save_tracks_to_txt(results: list, txt_path, conf=0.0):
    """

    Args:
        conf: элементы с conf менее указанной не сохраняются
        txt_path: текстовый файл для сохранения
        results: результат работы модели
    """
    with open(txt_path, 'w') as text_file:
        for track in results:
            if track[7] < conf:
                continue
            # print(frame_index, " ", box.id, ", cls = ", box.cls, ", ", box.xywhn)
            xywhn = track[1:5]
            # print(frame_index, " ", xywhn)
            bbox_w = xywhn[2]
            bbox_h = xywhn[3]
            bbox_left = xywhn[0]
            bbox_top = xywhn[1]
            track_id = int(track[5])
            cls = int(track[6])
            text_file.write(('%g ' * 8 + '\n') % (track[0], track_id, cls, bbox_left,
                                                  bbox_top, bbox_w, bbox_h, track[7]))


def yolo7_save_tracks_to_json(results: list, json_file, conf=0.0):
    """

    Args:
        conf: элементы с conf менее указанной не сохраняются
        json_file: json файл для сохранения
        results: результат работы модели
    """

    results_json = []

    for track in results:
        object_conf = track[7]
        if object_conf < conf:
            continue
        ltwhn = track[1:5]
        track_id = int(track[5])
        cls = int(track[6])
        frame_index = track[0]

        track = Track(ltwhn, cls, object_conf, frame_index, track_id)

        results_json.append(track)

    with open(json_file, "w") as write_file:
        write_file.write(json.dumps(results_json, indent=4, default=lambda o: o.__dict__))


def yolo_load_detections_from_txt(txt_path) -> DataFrame:
    """
    Получить дата фрейм из файла
    Parameters
    ----------
    txt_path
        Путь к файлу

    Returns
    -------
        Дата фрейм

    """
    if Path(txt_path).suffix == ".npy":
        return yolo_load_detections_from_npy(txt_path)

    try:
        if Path(txt_path).stat().st_size > 0:
            data_frame = pd.read_csv(txt_path, delimiter=" ", dtype=float, header=None)
        else:
            # если файл пустой, создаем пустой df, f nj pd.read_csv exception выдает
            data_frame = pd.DataFrame(dtype=float, columns=[0, 1, 2, 3, 4, 5, 6, 7])
        # data_frame = pd.DataFrame(df, columns=['frame', 'id', 'class',
        # 'bb_left', 'bb_top', 'bb_width', 'bb_height', 'conf'])
    except Exception as ex:
        print_exception(ex, f"yolo_load_detections_from_txt: '{str(txt_path)}'")
        data_frame = pd.DataFrame(dtype=float, columns=[0, 1, 2, 3, 4, 5, 6, 7])

    return data_frame


def yolo_load_detections_from_npy(txt_path) -> DataFrame:
    """
    Получить дата фрейм из файла в формате numpy
    Parameters
    ----------
    txt_path
        Путь к файлу

    Returns
    -------
        Дата фрейм

    """
    try:
        if Path(txt_path).stat().st_size > 0:
            # df = pd.read_csv(txt_path, delimiter=" ", dtype=float, header=None)

            all_boxes_and_shp = np.load(txt_path, allow_pickle=True)
            orig_shp = all_boxes_and_shp[0]  # Здесь формат
            width, height = orig_shp[1], orig_shp[0]
            all_boxes = all_boxes_and_shp[1]  # Здесь боксы

            tracks = []

            for item in all_boxes:
                left = item[0] / width
                top = item[1] / height

                width = (item[2] - item[0]) / width
                height = (item[3] - item[1]) / height

                frame_index, track_id, cls, conf = item[6], -1, item[5], item[4]

                # from bboxes - ndarray(x1, y1, x2, y2, conf, class, frame),
                # to [frame_index, track_id, cls, bbox_left, bbox_top, bbox_w, bbox_h, box.conf]

                tracks.append([frame_index, track_id, cls, left, top, width, height, conf])

            data_frame = pd.DataFrame(data=tracks, dtype=float, columns=[0, 1, 2, 3, 4, 5, 6, 7])

        else:
            # если файл пустой, создаем пустой df, f nj pd.read_csv exception выдает
            data_frame = pd.DataFrame(dtype=float, columns=[0, 1, 2, 3, 4, 5, 6, 7])
        # data_frame = pd.DataFrame(df, columns=['frame', 'id', 'class',
        # 'bb_left', 'bb_top', 'bb_width', 'bb_height', 'conf'])
    except Exception as ex:
        print_exception(ex, f"yolo_load_detections_from_npy: '{str(txt_path)}'")
        data_frame = pd.DataFrame(dtype=float, columns=[0, 1, 2, 3, 4, 5, 6, 7])

    return data_frame


if __name__ == '__main__':
    file_path = "D:\\AI\\2023\\Goup1\\78.npy"

    dff = yolo_load_detections_from_txt(file_path)
    print(dff)
