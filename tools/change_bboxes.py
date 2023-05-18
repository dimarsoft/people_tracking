"""
Модуль для изменения bbox
"""
from typing import Optional, Union

from tools.exception_tools import print_exception


def scale_bbox(bbox, scale: float):
    """
    Масштабирование bbox
    Args:
        bbox: bbox на замену
        scale (float): Масштаб
    Returns:
        bbox
    """
    x1_center = (bbox[:, [0]] + bbox[:, [2]]) / 2
    y1_center = (bbox[:, [1]] + bbox[:, [3]]) / 2

    scale /= 2

    width = abs(bbox[:, [0]] - bbox[:, [2]]) * scale
    height = abs(bbox[:, [1]] - bbox[:, [3]]) * scale

    bbox[:, [0]] = x1_center - width
    bbox[:, [2]] = x1_center + width

    bbox[:, [1]] = y1_center - height
    bbox[:, [3]] = y1_center + height

    return bbox


def change_bbox(bbox, change_bb: Union[bool, callable], file_id: Optional[str] = None,
                clone: bool = False):
    """
    Изменение bbox
    Parameters
    ----------

    bbox:
        bbox, который нужно изменить.

    change_bb:
        Менять/не менять bbox или  функция, меняющая bbox.

    file_id:
        номер файла.
    clone:
        флаг, указывающий на тоЮ что нужно ли делать копию bbox перед изменением.

    Returns
    -------
        Измененный bbox

    """
    if change_bb is None:
        return bbox

    if callable(change_bb):
        try:
            if clone:
                bbox = bbox.clone()
            return change_bb(bbox, file_id)
        except Exception as ex:
            print_exception(ex, "external change bbox")
            return bbox

    if isinstance(change_bb, float):
        if clone:
            bbox = bbox.clone()
        return scale_bbox(bbox, change_bb)

    if not isinstance(change_bb, bool):
        return bbox

    if not change_bb:
        return bbox

    if clone:
        bbox = bbox.clone()

    x_c = (bbox[:, [0]] + bbox[:, [2]]) / 2
    y_c = (bbox[:, [1]] + bbox[:, [3]]) / 2

    width = 10  # abs(bbox[:, [0]] - bbox[:, [2]]) / 4
    height = 10  # abs(bbox[:, [1]] - bbox[:, [3]]) / 4

    bbox[:, [0]] = x_c - width
    bbox[:, [2]] = x_c + width

    bbox[:, [1]] = y_c - height
    bbox[:, [3]] = y_c + height

    return bbox


def no_change_bbox(bbox, file_id: str):
    """
    Bbox возвращается без изменений

    Args:
        file_id(str): имя файла
        bbox (tensor): первые 4ре столбца x1 y1 x2 y2
        в абсолютных значениях картинки.
        Пока это ббоксы всех классов,
        сам класс находится в последнем столбце (-1, или индекс 5)
    """
    print(f"file = {file_id}")
    return bbox


def change_bbox_to_center(bbox, file_id: str):
    """
    Пример реализации бокса по центру 20/20

    Parameters
    ----------
    bbox
        bbox на входе
    file_id
        номер видео

    Returns
    -------
    измененный bbox

    """
    x_c = (bbox[:, [0]] + bbox[:, [2]]) / 2
    y_c = (bbox[:, [1]] + bbox[:, [3]]) / 2

    width = 10
    height = 10

    bbox[:, [0]] = x_c - width
    bbox[:, [2]] = x_c + width

    bbox[:, [1]] = y_c - height
    bbox[:, [3]] = y_c + height

    _ = file_id

    return bbox


def pavel_change_bbox(bbox, file_id: str):
    """
    Пример реализации от Павла (группа №1)

    Parameters
    ----------
    bbox
        bbox на входе
    file_id
        номер видео

    Returns
    -------
    измененный bbox

    """
    y_2 = bbox[:, [1]] + 150

    bbox[:, [3]] = y_2

    _ = file_id

    return bbox
