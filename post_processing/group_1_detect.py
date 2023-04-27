from pathlib import Path
from typing import Union

import gdown
import numpy as np
from ultralytics import YOLO

from configs import ROOT, WEIGHTS
from post_processing.group_1_tools import get_boxes, tracking_on_detect, get_men, get_count_men, get_count_vialotion
from post_processing.timur import get_camera
from tools.count_results import Deviation, Result
from tools.labeltools import get_status
from tools.resultools import results_to_dict
from trackers.multi_tracker_zoo import create_tracker


def group_1_detect(source,
                   model_path: Union[str, Path, None] = None,
                   tracker_config: Union[dict, Path, None] = None) -> dict:
    if tracker_config is None:
        tracker_config = ROOT / "trackers/ocsort/configs/ocsort_group1.yaml"

    if model_path is None:
        local_path = Path(WEIGHTS) / "yolo8_model_group1.pt"

        if not local_path.exists():
            url = 'https://drive.google.com/uc?id=1hszllwKkhl0M4o3meNDBkWONFvNP-NCF'
            gdown.download(url, str(local_path), quiet=False)

        model_path = str(local_path)

    # каждый раз инициализируем модель в колабе иначе выдает ошибочный результат
    model = YOLO(model_path)
    results, all_boxes, orig_shape = get_boxes(model.predict(source, stream=False, save=False))

    all_boxes_and_shp = np.array((orig_shape, all_boxes))

    ocsort_tracker = create_tracker("ocsort_v2", tracker_config)

    orig_shp = all_boxes_and_shp[0]  # Здесь формат
    all_boxes = all_boxes_and_shp[1]  # Здесь боксы

    # Отправляем боксы в трекинг + пробрасываем мимо трекинга каски и нетрекованные боксы людей
    out_boxes = tracking_on_detect(all_boxes, ocsort_tracker, orig_shp)

    # Смотрим у какого айди есть каски и жилеты (по порогу от доли кадров где был зафиксирован
    # айди человека + каска и жилет в его бб и без них)
    men = get_men(out_boxes)

    # здесь переназначаем айди входящий/выходящий (временное решение для MVP, надо думать над продом)
    men_clean, incoming1, exiting1 = get_count_men(men, orig_shp[1])

    # Здесь принимаем переназначенные айди смотрим нарушения,
    # а также повторно считаем входящих по дистанции, проверяем
    violation, incoming2, exiting2, df, clothing_helmet, clothing_unif = \
        get_count_vialotion(men_clean, orig_shp[1])
    deviations = []

    # 'helmet', 'uniform', 'first_frame', 'last_frame'

    for row in range(len(violation)):
        start_frame = violation["first_frame"].iloc[row]
        end_frame = violation['last_frame'].iloc[row]

        helmet = violation["helmet"].iloc[row]
        uniform = violation['uniform'].iloc[row]

        status = get_status(helmet == 0, uniform == 0)
        deviations.append(Deviation(int(start_frame), int(end_frame), status))

    results = Result(incoming2 + exiting2, incoming2, exiting2, deviations)
    results.file = str(source)

    results = results_to_dict(results)

    num, w, h, fps = get_camera(source)

    results["fps"] = fps

    return results


def group_1_detect_npy(source: Union[str, Path],
                       tracker_config: Union[dict, Path, None] = None) -> Result:
    """
    Трекинг по сохраненным файлам
    :param source: Путь к файлу
    :param tracker_config: Настройка трекера, если None, то будет использоваться в репы
    :return: Result
    """
    if tracker_config is None:
        tracker_config = ROOT / "trackers/ocsort/configs/ocsort_group1.yaml"

    # ocsort_v2 = OCSort, именно который использовала группа №1
    ocsort_tracker = create_tracker("ocsort_v2", tracker_config)

    all_boxes_and_shp = np.load(source, allow_pickle=True)
    orig_shp = all_boxes_and_shp[0]  # Здесь формат
    all_boxes = all_boxes_and_shp[1]  # Здесь боксы

    # Отправляем боксы в трекинг + пробрасываем мимо трекинга каски и нетрекованные боксы людей
    out_boxes = tracking_on_detect(all_boxes, ocsort_tracker, orig_shp)

    # Смотрим у какого айди есть каски и жилеты (по порогу от доли кадров где был зафиксирован
    # айди человека + каска и жилет в его бб и без них)
    men = get_men(out_boxes)

    # здесь переназначаем айди входящий/выходящий (временное решение для MVP, надо думать над продом)
    men_clean, incoming1, exiting1 = get_count_men(men, orig_shp[1])

    # Здесь принимаем переназначенные айди смотрим нарушения,
    # а также повторно считаем входящих по дистанции, проверяем
    violation, incoming2, exiting2, df, clothing_helmet, clothing_unif = \
        get_count_vialotion(men_clean, orig_shp[1])
    deviations = []

    # 'helmet', 'uniform', 'first_frame', 'last_frame'

    for row in range(len(violation)):
        start_frame = violation["first_frame"].iloc[row]
        end_frame = violation['last_frame'].iloc[row]

        helmet = violation["helmet"].iloc[row]
        uniform = violation['uniform'].iloc[row]

        status = get_status(helmet == 0, uniform == 0)
        deviations.append(Deviation(int(start_frame), int(end_frame), status))

    results = Result(incoming2 + exiting2, incoming2, exiting2, deviations)
    results.file = str(source)

    return results
