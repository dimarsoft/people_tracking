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
from utils.torch_utils import git_describe, date_modified


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

    glob_kwarg = {'barier': 358, 'tail_mark': False, 'tail': 200, 're_id_mark': False, 're_id_frame': 11,
                  'tail_for_count_mark': False, 'tail_for_count': 200, 'two_lines_buff_mark': False, 'buff': 40,
                  'go_men_forward': False, 'step': 45, 'height': 100}

    # каждый раз инициализируем модель в колабе иначе выдает ошибочный результат
    model = YOLO(model_path)
    all_boxes, orig_shape = get_boxes(
        model.predict(source, stream=True, save=False))

    all_boxes_and_shp = np.array((orig_shape, all_boxes))

    ocsort_tracker = create_tracker("ocsort_v2", tracker_config)

    orig_shp = all_boxes_and_shp[0]  # Здесь формат
    all_boxes = all_boxes_and_shp[1]  # Здесь боксы

    # Отправляем боксы в трекинг + пробрасываем мимо трекинга каски и нетрекованные боксы людей
    out_boxes = tracking_on_detect(
        all_boxes, ocsort_tracker, orig_shp, **glob_kwarg)

    # Смотрим у какого айди есть каски и жилеты (по порогу от доли кадров где был зафиксирован
    # айди человека + каска и жилет в его бб и без них)
    men = get_men(out_boxes)

    # здесь переназначаем айди входящий/выходящий
    men_clean = get_count_men(men, orig_shp[0], **glob_kwarg)

    # Здесь принимаем переназначенные айди смотрим нарушения,
    # а также повторно считаем входящих по дистанции, проверяем
    violation, incoming, exiting, clothing_helmet, clothing_unif = \
        get_count_vialotion(men_clean, orig_shp[0], **glob_kwarg)
    deviations = []

    # 'helmet', 'uniform', 'first_frame', 'last_frame'

    for row in range(len(violation)):
        start_frame = violation["first_frame"].iloc[row]
        end_frame = violation['last_frame'].iloc[row]

        helmet = violation["helmet"].iloc[row]
        uniform = violation['uniform'].iloc[row]

        status = get_status(helmet == 0, uniform == 0)
        deviations.append(Deviation(int(start_frame), int(end_frame), status))

    results = Result(incoming + exiting, incoming, exiting, deviations)
    results.file = str(source)

    results = results_to_dict(results)

    num, w, h, fps = get_camera(source)

    results["fps"] = fps

    return results


def print_version():
    git_info = git_describe()

    if git_info is None:
        git_info = f"{date_modified()}"
    else:
        git_info = f"git: {git_info}, {date_modified()}"

    version = f'Version: {git_info}'
    print(f"group_1_detect_npy. {version}")


def group_1_detect_npy(source: Union[str, Path],
                       tracker_config: Union[dict, Path, None] = None) -> Result:
    """
    Трекинг по сохраненным файлам
    :param source: Путь к файлу
    :param tracker_config: Настройка трекера, если None, то будет использоваться в репы
    :return: Result
    """

    print_version()

    glob_kwarg = {'barier': 358, 'tail_mark': False, 'tail': 200, 're_id_mark': False, 're_id_frame': 11,
                  'tail_for_count_mark': False, 'tail_for_count': 200, 'two_lines_buff_mark': False, 'buff': 40,
                  'go_men_forward': False, 'step': 45, 'height': 100}

    if tracker_config is None:
        tracker_config = ROOT / "trackers/ocsort/configs/ocsort_group1.yaml"

    # ocsort_v2 = OCSort, именно который использовала группа №1
    ocsort_tracker = create_tracker("ocsort_v2", tracker_config)

    all_boxes_and_shp = np.load(source, allow_pickle=True)
    orig_shp = all_boxes_and_shp[0]  # Здесь формат
    all_boxes = all_boxes_and_shp[1]  # Здесь боксы

    # Отправляем боксы в трекинг + пробрасываем мимо трекинга каски и нетрекованные боксы людей
    out_boxes = tracking_on_detect(
        all_boxes, ocsort_tracker, orig_shp, **glob_kwarg)

    # Смотрим у какого айди есть каски и жилеты (по порогу от доли кадров где был зафиксирован
    # айди человека + каска и жилет в его бб и без них)
    men = get_men(out_boxes)

    # здесь переназначаем айди входящий/выходящий
    men_clean = get_count_men(men, orig_shp[0], **glob_kwarg)

    # Здесь принимаем переназначенные айди смотрим нарушения,
    # а также повторно считаем входящих по дистанции, проверяем
    violation, incoming, exiting, clothing_helmet, clothing_unif = \
        get_count_vialotion(men_clean, orig_shp[0], **glob_kwarg)
    deviations = []

    # 'helmet', 'uniform', 'first_frame', 'last_frame'

    for row in range(len(violation)):
        start_frame = violation["first_frame"].iloc[row]
        end_frame = violation['last_frame'].iloc[row]

        helmet = violation["helmet"].iloc[row]
        uniform = violation['uniform'].iloc[row]

        status = get_status(helmet == 0, uniform == 0)
        deviations.append(Deviation(int(start_frame), int(end_frame), status))

    results = Result(incoming + exiting, incoming, exiting, deviations)
    results.file = str(source)

    return results
